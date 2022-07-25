import numpy as np
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from peakdetect.utils.loss import compute_loss
from peakdetect.models import load_model
from peakdetect.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info, rescale_boxes

class EDPeakDector(pl.LightningModule):
    def __init__(self, 
                 input_dp_size,
                 class_names_path,
                 model_config_path,
                 conf_thres,
                 iou_thres,
                 nms_thres,):
        super().__init__()
        self.save_hyperparameters()
        self.input_dp_size = input_dp_size
        self.class_names = load_classes(class_names_path)
        self.model = load_model(model_config_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.nms_thres = nms_thres
        
    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/precision": 0, "hp/recall": 0, "hp/f1":0})

    def training_step(self, train_batch, batch_idx):
        imgs, targets, _, _ = train_batch
        imgs = imgs.to(dtype=torch.float)

        outputs = self.model(imgs)
        loss, loss_components = compute_loss(outputs, targets, self.model)
        self.model.seen += imgs.size(0)
        
        metrics = {
            'loss': loss,
            'IoU_loss': float(loss_components[0]),
            'Object_loss': float(loss_components[1]),
            'Class_loss': float(loss_components[2]),
        }
        self.log('IoU_loss', metrics['IoU_loss'], prog_bar=True, on_step=True)
        self.log('Object_loss', metrics['Object_loss'], prog_bar=True, on_step=True)
        self.log('Class_loss', metrics['Class_loss'], prog_bar=True, on_step=True)

        return metrics

    def training_epoch_end(self, training_step_outputs):
        if(self.current_epoch==1):
            sample_img=torch.rand((1,1,self.input_dp_size,self.input_dp_size))
            self.logger.experiment.add_graph(self.model(),sample_img)

        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        avg_IoU_loss = torch.stack([x['IoU_loss'] for x in training_step_outputs]).mean()
        avg_Object_loss = torch.stack([x['Object_loss'] for x in training_step_outputs]).mean()
        avg_Class_loss = torch.stack([x['Class_loss'] for x in training_step_outputs]).mean()

        self.logger.experiment.add_scalar('Avg_loss', avg_loss,avg_IoU_loss.current_epoch)
        self.logger.experiment.add_scalar('avg_IoU_loss', avg_loss,self.current_epoch)
        self.logger.experiment.add_scalar('avg_Object_loss', avg_Object_loss,self.current_epoch)
        self.logger.experiment.add_scalar('avg_Class_loss', avg_Class_loss,self.current_epoch)


    def validation_step(self, val_batch, batch_idx):
        imgs, targets, _, _  = val_batch
        imgs = imgs.to(dtype=torch.float)
        # Extract labels
        labels = targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= imgs.size(3)

        outputs = self.model(imgs)
        outputs = non_max_suppression(outputs, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

        sample_metrics = get_batch_statistics(outputs, targets, iou_threshold=self.iou_thres)

        return {'imgs':imgs,
                'targets':targets,
                'outputs':outputs,
                'labels':labels,
                'sample_metrics': sample_metrics}
    
    def validation_epoch_end(self, valid_step_outputs):
        # Concatenate sample statistics
        labels=[]
        sample_metrics=[]
        for batch in valid_step_outputs:
            labels+=batch['labels']
            sample_metrics+=batch['sample_metrics']

        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

        precision, recall, AP, f1, ap_class = ap_per_class(
            true_positives, pred_scores, pred_labels, labels)
        
        self.logger.experiment.add_scalar('precision', precision.mean(), self.current_epoch)
        self.logger.experiment.add_scalar('recall', recall.mean(), self.current_epoch)
        self.logger.experiment.add_scalar('f1', f1.mean(), self.current_epoch) 

        self.log("hp/precision", precision.mean())
        self.log("hp/recall", recall.mean())
        self.log("hp/f1", f1.mean())    
        
        # plot validation results
        last_batch = valid_step_outputs[-1]
        fig = _plot_evaluation(last_batch['imgs'], last_batch['targets'], last_batch['outputs'], self.class_names)
        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        eval_result = np.array(fig.canvas.renderer.buffer_rgba())
        eval_result = np.moveaxis(eval_result[:,:,:3],2,0)
        self.logger.experiment.add_image(f'eval_check_{self.current_epoch}', eval_result)

        ap_dict = {}
        for i, c in enumerate(ap_class):
            ap_dict[self.class_names[c]] = AP[i]
        self.log_dict(ap_dict, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        if (self.model.hyperparams['optimizer'] in [None, "adam"]):
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.model.hyperparams['learning_rate'],
                weight_decay=self.model.hyperparams['decay'],
            )
        elif (self.model.hyperparams['optimizer'] == "sgd"):
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.model.hyperparams['learning_rate'],
                weight_decay=self.model.hyperparams['decay'],
                momentum=self.model.hyperparams['momentum'])
        else:
            print("Unknown optimizer. Please choose between (adam, sgd).")

        return optimizer
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        #                                                         mode='max', 
        #                                                         factor=0.1, 
        #                                                         patience=5, 
        #                                                         threshold=0.01, 
        #                                                         verbose=True)

    #     return {
    #     "optimizer": optimizer,
    #     "lr_scheduler": {
    #         "scheduler": scheduler,
    #         "monitor": "precision",
    #     },
    # }

    def detect_sigle_dp(self, img, targets):
        self.eval().cuda()

        img = img.unsqueeze(0).to(device='cuda', dtype=torch.float)
        outputs = self.model(img)
        outputs = non_max_suppression(outputs, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img.size(3)
        targets = targets.cpu()

        detections_pred = outputs[0].cpu()
        detections_target = targets.cpu()
        num_detections_target = detections_target.size(0)
        detections_target = torch.cat((detections_target, torch.ones(num_detections_target,1)),axis=1)
        detections_target = detections_target[:,[2,3,4,5,6,1]]

        # colors
        unique_labels = torch.cat((detections_pred,detections_target),axis=0)[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
        bbox_colors = colors #random.sample(colors, n_cls_preds)

        fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=150)
        for i in range(3):
            if i==0:
                axs[i].imshow(img.squeeze().cpu().detach().numpy())

            else:
                # model prediction
                if i==1:
                    axs[i].imshow(img.squeeze().cpu().detach().numpy())
                    detections = detections_pred.to("cpu")

                # ground truth
                elif i==2:
                    axs[i].imshow(img.squeeze().cpu().detach().numpy())
                    detections = detections_target.cpu()


                for x1, y1, x2, y2, conf, cls_pred in detections:

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    axs[i].add_patch(bbox)
                    # Add label
                    if i==2:
                        axs[i].text(
                            x1,#-0.5*box_w,
                            y2,
                            fontsize=5,
                            s=self.class_names[int(cls_pred)],
                            color="white",
                            verticalalignment="top",
                            horizontalalignment="left",
                            bbox={"color": color, "pad": 0})
                    if i==1:
                        axs[i].text(
                            x1,#,
                            y2,
                            fontsize=5,
                            s=self.class_names[int(cls_pred)]+f'\n{conf*100:.1f}%',
                            color="white",
                            verticalalignment="top",
                            horizontalalignment="left",
                            bbox={"color": color, "pad": 0})

            axs[i].axis("off")
            axs[i].xaxis.set_major_locator(NullLocator())
            axs[i].yaxis.set_major_locator(NullLocator())

        fig.subplots_adjust(wspace=0.1, hspace=0.00)
        plt.show()


def _plot_evaluation(imgs, targets, outputs, classes):
    imgs = imgs.to("cpu")
    targets = targets.to("cpu")

    img_ids = random.sample(range(imgs.size(0)),4)
    fig, axs = plt.subplots(2,4, figsize=(6,3), dpi=150)
    for i in range(2):
        for j in range(4):
            img_id = img_ids[j]
            axs[i,j].imshow(imgs[img_id].squeeze().numpy())

            # model prediction
            detections_pred = outputs[img_id]

            # ground truth
            detections_target = targets[np.where(targets.numpy()[:,0]==img_id)]
            num_detections_target = detections_target.size(0)
            detections_target = torch.cat((detections_target, torch.ones(num_detections_target,1)),axis=1)
            detections_target = detections_target[:,[2,3,4,5,6,1]]
            
            if i==0: 
                detections = detections_pred.to("cpu")
            else: 
                detections = detections_target.to("cpu")

            # colors
            unique_labels = torch.cat((detections_pred,detections_target),axis=0)[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            # Bounding-box colors
            cmap = plt.get_cmap("tab20b")
            colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
            bbox_colors = colors #random.sample(colors, n_cls_preds)

            for x1, y1, x2, y2, conf, cls_pred in detections:

                # print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                axs[i,j].add_patch(bbox)
                # Add label
                axs[i,j].text(
                    x1,#-0.5*box_w,
                    y2,
                    fontsize=4,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox={"color": color, "pad": 0})
            axs[i,j].axis("off")
            axs[i,j].xaxis.set_major_locator(NullLocator())
            axs[i,j].yaxis.set_major_locator(NullLocator())

    fig.subplots_adjust(wspace=0.0, hspace=0.05)
    # plt.show()
    return fig


def detect_sigle_dp(classes, img, model, targets):
    model.eval().cuda()

    img = img.unsqueeze(0).to(device='cuda', dtype=torch.float)
    outputs = model(img)
    outputs = non_max_suppression(outputs, conf_thres=model.conf_thres, iou_thres=model.iou_thres)

    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    targets[:, 2:] *= img.size(3)
    targets = targets.cpu()

    detections_pred = outputs[0].cpu()
    detections_target = targets.cpu()
    num_detections_target = detections_target.size(0)
    detections_target = torch.cat((detections_target, torch.ones(num_detections_target,1)),axis=1)
    detections_target = detections_target[:,[2,3,4,5,6,1]]

    # colors
    unique_labels = torch.cat((detections_pred,detections_target),axis=0)[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = colors #random.sample(colors, n_cls_preds)

    fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=150)
    for i in range(3):
        if i==0:
            axs[i].imshow(img.squeeze().cpu().detach().numpy())

        else:
            # model prediction
            if i==1:
                axs[i].imshow(img.squeeze().cpu().detach().numpy())
                detections = detections_pred.to("cpu")

            # ground truth
            elif i==2:
                axs[i].imshow(img.squeeze().cpu().detach().numpy())
                detections = detections_target.cpu()


            for x1, y1, x2, y2, conf, cls_pred in detections:

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                axs[i].add_patch(bbox)
                # Add label
                if i==2:
                    axs[i].text(
                        x1,#-0.5*box_w,
                        y2,
                        fontsize=5,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        horizontalalignment="left",
                        bbox={"color": color, "pad": 0})
                if i==1:
                    axs[i].text(
                        x1,#,
                        y2,
                        fontsize=5,
                        s=classes[int(cls_pred)]+f'\n{conf*100:.1f}%',
                        color="white",
                        verticalalignment="top",
                        horizontalalignment="left",
                        bbox={"color": color, "pad": 0})

        axs[i].axis("off")
        axs[i].xaxis.set_major_locator(NullLocator())
        axs[i].yaxis.set_major_locator(NullLocator())

    fig.subplots_adjust(wspace=0.1, hspace=0.00)
    plt.show()

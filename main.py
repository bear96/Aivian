from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time
import deeplake
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.datasets import ImageFolder
from transformers import ViTConfig
import torchvision.transforms as transforms
from PIL import Image

from datetime import timedelta

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import ResNet, ResNet152, VisionTransformer, Efficient_Net
from scheduler import WarmupLinearSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pkl" % args.name)
    checkpoint = {'model': model_to_save.state_dict()}
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    if args.model == "ResNet":
        model = ResNet(args.num_classes)
    elif args.model == "ResNet152":
        model = ResNet152(args.num_classes)
    elif args.model == "Transformer":
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        model = VisionTransformer(config, args.num_classes)
    elif args.model == 'EfficientNet':
        model = Efficient_Net(args.num_classes)
    else:
        print("Invalid choice.")
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def valid(args, model, writer, test_loader, global_step, cls_to_idx=None):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        if args.use_deeplake:
            x = batch['images']
            y = batch['labels']
            y = torch.tensor([torch.tensor(cls_to_idx[cls.item()]) for cls in y])
        else:
            x,y = batch
        x = x.to(args.device)
        y = y.to(args.device)
        
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    val_accuracy = accuracy.detach().cpu().numpy()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)
        
    return val_accuracy

def train(args, model, train_loader,test_loader,cls_to_idx=None):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    tr_loss = []
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    loss_fct = torch.nn.CrossEntropyLoss()

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=args.learning_rate)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            if args.use_deeplake:
                x = batch['images']
                y = batch['labels']
                y = torch.tensor([torch.tensor(cls_to_idx[cls.item()]) for cls in y])
            else: 
                x,y = batch
                
            x = x.to(args.device)
            y = y.to(args.device)

            
            logits = model(x)
            loss = loss_fct(logits,y)
            loss = loss.mean()
            tr_loss.append(loss.item())

            preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0:
                    with torch.no_grad():
                        accuracy = valid(args, model, writer, test_loader, global_step,cls_to_idx)
                    if best_acc < accuracy:
                            save_model(args, model)
                            best_acc = accuracy
                    logger.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step % t_total == 0:
                    break
        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args.device)
        train_accuracy = accuracy.detach().cpu().numpy()
        logger.info("train accuracy so far: %f" % train_accuracy)
        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))
    with open(r"train_loss.txt","wb") as f:
        for item in tr_loss:
            fp.write("%s\n" % item)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required = True , default= "Aivian",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--model", choices=["ResNet","ResNet152","Efficient_Net","Transformer"],
                        default="ResNet",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--train_dir", default = '/home/aritram21/Aivian/Cropped_Data/output/train', type=str, help="Directory of the dataset for training.")
    parser.add_argument("--valid_dir", default = '/home/aritram21/Aivian/Cropped_Data/output/val', type=str, help="Directory of the dataset for validation.")
    parser.add_argument("--test_dir", default = '/home/aritram21/Aivian/Cropped_Data/output/test', type=str, help="Directory of the dataset for testing.")
    parser.add_argument("--output_dir", default="./Aivian_ResNet_output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=500, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--num_classes", default = 555, type = int, help = "Number of classes in the dataset.")
    parser.add_argument("--learning_rate", default=5.5e-5, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--image_resolution", type=int, default=448,
                        help="image resolution of input image, default 448x448. For ViT use 224.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--use_deeplake', action = "store_true", default = False, help = "Whether to use deeplake or raw images.")

    args = parser.parse_args()
    n = args.image_resolution
    if args.use_deeplake:
        train_transform = transforms.Compose([transforms.ToPILImage(mode="RGB"),transforms.Resize((600, 600), transforms.InterpolationMode.BILINEAR),
                                        transforms.RandomCrop((n, n)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.ToPILImage(mode = "RGB"), transforms.Resize((600, 600), transforms.InterpolationMode.BILINEAR),
                                        transforms.CenterCrop((n, n)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        trainset = deeplake.load('hub://activeloop/nabirds-dataset-train')
        testset = deeplake.load('hub://activeloop/nabirds-dataset-val')
        train_loader = trainset.pytorch(num_workers=0, batch_size=args.train_batch_size, transform={
                        'images': train_transform, 'labels': None}, shuffle = True, decode_method = {"images":"numpy"})
        test_loader = trainset.pytorch(num_workers=0, batch_size=args.eval_batch_size, transform={
                        'images': test_transform, 'labels': None}, shuffle = False, decode_method = {"images":"numpy"})
        deeplake_classes = set([int(cls) for cls in trainset.labels.numpy()])
        cls_to_idx = {}
        i=0
        for cls in deeplake_classes:
            cls_to_idx[cls] = i
            i+=1
    else:
        train_transform = transforms.Compose([transforms.Resize((n, n)),  #transforms.Resize((600, 600), transforms.InterpolationMode.BILINEAR)
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((n, n)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = ImageFolder(args.train_dir, transform = train_transform)
        testset = ImageFolder(args.valid_dir, transform = test_transform)
        
        train_sampler = RandomSampler(trainset)
        test_sampler = SequentialSampler(testset)
        train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=1,
                              drop_last=True,
                              pin_memory=True)
        test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=1,
                             pin_memory=True)
        
        cls_to_idx = None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), False))

    # Set seed
    set_seed(args)
    args, model = setup(args)
    # Training
    train(args, model,train_loader,test_loader,cls_to_idx)

if __name__ == "__main__":
    main()

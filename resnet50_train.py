import os
import argparse

from tqdm import tqdm
from datetime import datetime

import torch
from torchvision.datasets import ImageNet
from torchvision.models import resnet50
from torchvision import transforms
from torch.profiler import profile, record_function, ProfilerActivity, schedule

def main():

    parser = argparse.ArgumentParser(
                    prog='ResNet50 ImageNet',
                    description='Trains ResNet50 on the ImageNet ILSVRC2012 dataset.',
                    epilog='Text at the bottom of help')
    parser.add_argument('batch_size', type=int, default=64,
                        help='Batch size used for inference.')
    parser.add_argument('--nWorkers', type=int, default=1,
                        help='number of cores to use in data loading')
    parser.add_argument('--nEpochs', type=int, default=1,
                        help='number epochs to train')
    parser.add_argument('--skip_first', type=int, default=1,
                        help='number of steps to skip for profiling')
    parser.add_argument('--warmup', type=int, default=1,
                        help='number of warmup steps for profiling')
    parser.add_argument('--active', type=int, default=1,
                        help='number of steps to record skip for profiling')
    parser.add_argument('--compile', action='store_true',
                        help='Compile the model and optimizer before evaluation.')

    args = parser.parse_args()

    slurm_tmpidir = os.environ['SLURM_TMPDIR']
    image_net_path = os.path.join(slurm_tmpidir, 'imagenet')

    # load data
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    


    imagenet_train_data = ImageNet(image_net_path, split='train', transform=transform)
    train_dataloader = torch.utils.data.DataLoader(imagenet_train_data,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.nWorkers)

    model = resnet50()
    if args.compile:
        model = torch.compile(model)

    model.cuda()

    my_schedule = schedule(
    skip_first=args.skip_first,
    wait=0,
    warmup=args.warmup,
    active=args.active,
    repeat=1)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True,
                 with_stack=True,
                 schedule=my_schedule) as prof:

        for epoch in range(args.nEpochs):
            print(f'EPOCH {epoch + 1}:')

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            running_loss = 0.

            with record_function('training'):

                for i, data in enumerate(tqdm(train_dataloader)):
                    # Every data instance is an input + label pair
                    inputs, labels = data

                    # Zero your gradients for every batch!
                    optimizer.zero_grad()

                    # Make predictions for this batch
                    outputs = model(inputs.cuda())

                    # Compute the loss and its gradients
                    loss = loss_fn(outputs, labels.cuda())
                    loss.backward()

                    # Adjust learning weights
                    optimizer.step()

                    # Gather data and report
                    running_loss += loss.item()
                    if i > 98:
                        avg_loss = running_loss / 100 # loss per batch
                        print(f'  batch {i + 1} loss: {avg_loss}')
                        running_loss = 0.
                        break
                    
                    prof.step()

    # print profiles
    print("GPU sorted stats:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total",
                                                             row_limit=10))

    print("\nCPU sorted stats:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print("\nCPU memory profile:")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    compiled = '_compiled' if args.compile else ''
    prof.export_chrome_trace('/path/to/traces/' 
        + f'train_batch_{args.batch_size}_nworkers_{args.nWorkers}{compiled}_trace.json')
    prof.export_memory_timeline('/path/to/traces/'
        + f'train_batch_{args.batch_size}_nworkers_{args.nWorkers}_memory'
        + '.html',
        device='cuda:0')
    # prof.export_memory_timeline('/path/to/traces/'
    #     + f'train_batch_{args.batch_size}_nworkers_{args.nWorkers}_memory'
    #     + '.json',
    #     device='cuda:0')

if __name__ == '__main__':
    main()

import os
import argparse
import torch
import pickle
from tqdm import tqdm
from torchvision.datasets import ImageNet
from torchvision.models import resnet50
from torchvision import transforms
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

def main():

    parser = argparse.ArgumentParser(
                    prog='ResNet50 ImageNet',
                    description='Trains ResNet50 on the ImageNet ILSVRC2012 dataset.',
                    epilog='Text at the bottom of help')
    parser.add_argument('batch_size', type=int, default=64,
                        help='Batch size used for inference.')
    parser.add_argument('--nWorkers', type=int, default=1,
                        help='number of cores to use in data loading')
    parser.add_argument('--compile', action='store_true',
                        help='Compile the model before evaluation.')

    args = parser.parse_args()

    slurm_tmpidir = os.environ['SLURM_TMPDIR']
    image_net_path = os.path.join(slurm_tmpidir, 'imagenet')

    # load data
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    imagenet_val_data = ImageNet(image_net_path, split='val', transform=val_transform)
    val_dataloader = torch.utils.data.DataLoader(imagenet_val_data,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.nWorkers)

    model = resnet50(weights="IMAGENET1K_V2")
    if args.compile:
        model = torch.compile(model)

    model.eval().cuda()

    my_schedule = schedule(skip_first=10,
                           wait=5,
                           warmup=5,
                           active=5,
                           repeat=1)
    trace_handler = tensorboard_trace_handler(dir_name='/home/c7wilson/project/pytorch_profiler_talk/traces/',
                                              use_gzip=True)

    correct = 0
    total = 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True,
                 with_stack=True,
                 schedule=my_schedule,
                 on_trace_ready=trace_handler) as prof:
        with torch.no_grad():
            for x, y in tqdm(val_dataloader):
                with record_function('model_inference'):
                    y_pred = model(x.cuda())
                correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
                total += len(y)
                prof.step()

    # print profiles
    print("GPU profiles:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total",
                                                             row_limit=10))

    print("\nCPU profile:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print("\nCPU memory profile:")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    compiled = '_compiled' if args.compile else ''
    prof.export_chrome_trace(
        f'/home/c7wilson/project/pytorch_profiler_talk/traces/' 
        + f'basic_inference_batch_{args.batch_size}_nworkers_{args.nWorkers}{compiled}_trace.json')
    prof.export_memory_timeline(f'/home/c7wilson/project/pytorch_profiler_talk/traces/'
        + f'basic_inference_batch_{args.batch_size}_nworkers_{args.nWorkers}{compiled}_memory'
        + '.html',
        device='cuda:0')
    
    save_name = f'profile_{args.batch_size}_nworkers_{args.nWorkers}{compiled}.pkl'
    with open ('/home/c7wilson/project/pytorch_profiler_talk/traces/' + save_name, 'wb') as file:
        pickle.dump(prof, file)

    print(correct / total)

if __name__ == '__main__':
    main()

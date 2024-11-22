#! /usr/bin/env python3
import argparse
import os
import torch
from hyperpyyaml import load_hyperpyyaml


torch.set_grad_enabled(False)


parser = argparse.ArgumentParser(description="ASV model tracing")
parser.add_argument('config_path', type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--fuse', default=False, action='store_true', help='True to fuse components of the model')
parser.add_argument('--show', default=False, action='store_true', help='True to print the model to the terminal')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--output_path', required=False, type=str)
args = parser.parse_args()


if __name__ == '__main__':
    with open(args.config_path, 'r') as f:
        yaml_strings = f.read()
        hparams = load_hyperpyyaml(yaml_strings)

    model = hparams['model']
    state_dict = torch.load(hparams['ckpt_path'], map_location=args.device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device).eval()
    # model.featCal = torch.nn.Identity()

    # example_input_array = torch.rand(1, 1, hparams['n_mels'], 301, device=args.device)
    example_input_array = torch.rand(1, 16000, device=args.device)
    reference_output_array = model(example_input_array)

    if args.fuse:  # Fuse the batch norm modules where possible.
        print('Fusing model')
        model.apply(lambda x: x.fuse_modules() if hasattr(x, 'fuse_modules') and callable(x.fuse_modules) else None)
        error = model(example_input_array).sub(reference_output_array).square().mean().sqrt()
        assert error < 1e-6, f"error should be less than 1e-6 but is {error}"
        print(f'MSE after fusion is {error}')

    if args.show:  # Print the model to the terminal
        print(model)

    # Trace the model.
    print('Tracing model')
    model = torch.jit.trace(model, example_input_array)
    error = model(example_input_array).sub(reference_output_array).square().mean().sqrt()
    print(f'MSE after tracing is {error}')
    # Save the model to the file-system.
    output_filename = f'{os.path.splitext(os.path.basename(args.config_path))[0]}.pt'
    print(f'Saving model to {output_filename}')
    model.save(output_filename)

#!/usr/bin/env python3

import torch
from timer import Timer


@torch.no_grad()
def compute_time_eval(model, im_size, batch_size):
    """Computes precise model forward test time using dummy data."""
    model.eval()
    # Generate a dummy mini-batch and copy data to GPU
    inputs = torch.zeros(batch_size, 3, im_size, im_size).cuda(non_blocking=False)
    # Compute precise forward pass time
    timer = Timer()
    total_iter = 1100

    # Run.
    for cur_iter in range(total_iter):
        # Reset the timers after the warmup phase
        if cur_iter == 100:
            timer.reset()
        # Forward
        timer.tic()
        model(inputs)
        torch.cuda.synchronize()
        timer.toc()
    return timer.average_time


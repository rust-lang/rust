Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
-   triton.testing
    -   triton.testing.Benchmark
    -   triton.testing.do_bench
    -   triton.testing.do_bench_cudagraph
        -   `do_bench_cudagraph()`
    -   triton.testing.perf_report
    -   triton.testing.assert_close
-   Triton Semantics

Triton MLIR Dialects

-   Triton MLIR Dialects and Ops

Programming Guide

-   Introduction
-   Related Work
-   Debugging Triton

Triton

-   
-   triton.testing
-   triton.testing.do_bench_cudagraph
-   View page source

------------------------------------------------------------------------

# triton.testing.do_bench_cudagraph¶

triton.testing.do_bench_cudagraph(*fn*, *rep=20*, *grad_to_none=None*, *quantiles=None*, *return_mode=\'mean\'*)¶

:   Benchmark the runtime of the provided function.

    Parameters:

    :   -   **fn** (*Callable*) -- Function to benchmark

        -   **rep** (*int*) -- Repetition time (in ms)

        -   **grad_to_none** (*torch.tensor,* *optional*) -- Reset the gradient of the provided tensor to None

        -   **return_mode** (*str*) -- The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

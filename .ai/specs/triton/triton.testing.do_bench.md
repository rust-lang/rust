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
        -   `do_bench()`
    -   triton.testing.do_bench_cudagraph
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
-   triton.testing.do_bench
-   View page source

------------------------------------------------------------------------

# triton.testing.do_bench¶

triton.testing.do_bench(*fn*, *warmup=25*, *rep=100*, *grad_to_none=None*, *quantiles=None*, *return_mode=\'mean\'*)¶

:   Benchmark the runtime of the provided function. By default, return the median runtime of `fn` along with the 20-th and 80-th performance percentile.

    Parameters:

    :   -   **fn** (*Callable*) -- Function to benchmark

        -   **warmup** (*int*) -- Warmup time (in ms)

        -   **rep** (*int*) -- Repetition time (in ms)

        -   **grad_to_none** (*torch.tensor,* *optional*) -- Reset the gradient of the provided tensor to None

        -   **quantiles** (*list\[float\],* *optional*) -- Performance percentile to return in addition to the median.

        -   **return_mode** (*str*) -- The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

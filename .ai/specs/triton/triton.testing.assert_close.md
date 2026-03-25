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
    -   triton.testing.perf_report
    -   triton.testing.assert_close
        -   `assert_close()`
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
-   triton.testing.assert_close
-   View page source

------------------------------------------------------------------------

# triton.testing.assert_close¶

triton.testing.assert_close(*x*, *y*, *atol=None*, *rtol=None*, *err_msg=\'\'*)¶

:   Asserts that two inputs are close within a certain tolerance.

    Parameters:

    :   -   **x** (*scala,* *list,* *numpy.ndarray, or* *torch.Tensor*) -- The first input.

        -   **y** (*scala,* *list,* *numpy.ndarray, or* *torch.Tensor*) -- The second input.

        -   **atol** (*float,* *optional*) -- The absolute tolerance. Default value is 1e-2.

        -   **rtol** (*float,* *optional*) -- The relative tolerance. Default value is 0.

        -   **err_msg** (*str*) -- The error message to use if the assertion fails.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

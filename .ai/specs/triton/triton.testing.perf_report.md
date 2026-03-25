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
        -   `perf_report()`
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
-   triton.testing.perf_report
-   View page source

------------------------------------------------------------------------

# triton.testing.perf_report¶

triton.testing.perf_report(*benchmarks*)¶

:   Mark a function for benchmarking. The benchmark can then be executed by using the `.run` method on the return value.

    Parameters:

    :   **benchmarks** (List of `Benchmark`) -- Benchmarking configurations.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

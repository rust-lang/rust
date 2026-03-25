Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
    -   Programming Model
    -   Creation Ops
    -   Shape Manipulation Ops
    -   Linear Algebra Ops
    -   Memory/Pointer Ops
    -   Indexing Ops
    -   Math Ops
    -   Reduction Ops
    -   Scan/Sort Ops
        -   triton.language.associative_scan
        -   triton.language.cumprod
        -   triton.language.cumsum
        -   triton.language.histogram
            -   `histogram()`
        -   triton.language.sort
        -   triton.language.gather
    -   Atomic Ops
    -   Random Number Generation
    -   Iterators
    -   Inline Assembly
    -   Compiler Hint Ops
    -   Debug Ops
-   triton.testing
-   Triton Semantics

Triton MLIR Dialects

-   Triton MLIR Dialects and Ops

Programming Guide

-   Introduction
-   Related Work
-   Debugging Triton

Triton

-   
-   triton.language
-   triton.language.histogram
-   View page source

------------------------------------------------------------------------

# triton.language.histogram¶

triton.language.histogram(*input*, *num_bins*, *mask=None*, *\_semantic=None*, *\_generator=None*)¶

:   computes an histogram based on input tensor with num_bins bins, the bins have a width of 1 and start at 0.

    Parameters:

    :   -   **input** (*Tensor*) -- the input tensor

        -   **num_bins** (*int*) -- number of histogram bins

        -   **mask** (Block of triton.int1, optional) -- if mask\[idx\] is false, exclude input\[idx\] from histogram

    This function can also be called as a member function on `tensor`, as `x.histogram(...)` instead of `histogram(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

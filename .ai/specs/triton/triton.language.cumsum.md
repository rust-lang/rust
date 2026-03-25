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
            -   `cumsum()`
        -   triton.language.histogram
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
-   triton.language.cumsum
-   View page source

------------------------------------------------------------------------

# triton.language.cumsum¶

triton.language.cumsum(*input*, *axis=0*, *reverse=False*, *dtype: constexpr \| None = None*)¶

:   Returns the cumsum of all elements in the `input` tensor along the provided `axis`

    Parameters:

    :   -   **input** (*Tensor*) -- the input values

        -   **axis** (*int*) -- the dimension along which the scan should be done

        -   **reverse** (*bool*) -- if true, the scan is performed in the reverse direction

        -   **dtype** (*tl.dtype*) -- the desired data type of the returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. If not specified, small integer types (\< 32 bits) are upcasted to prevent overflow. Note that `tl.bfloat16` inputs are automatically promoted to `tl.float32`.

    This function can also be called as a member function on `tensor`, as `x.cumsum(...)` instead of `cumsum(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

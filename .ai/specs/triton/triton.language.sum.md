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
        -   triton.language.argmax
        -   triton.language.argmin
        -   triton.language.max
        -   triton.language.min
        -   triton.language.reduce
        -   triton.language.sum
            -   `sum()`
        -   triton.language.xor_sum
    -   Scan/Sort Ops
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
-   triton.language.sum
-   View page source

------------------------------------------------------------------------

# triton.language.sum¶

triton.language.sum(*input*, *axis=None*, *keep_dims=False*, *dtype: constexpr \| None = None*)¶

:   Returns the sum of all elements in the `input` tensor along the provided `axis`

    Parameters:

    :   -   **input** (*Tensor*) -- the input values

        -   **axis** (*int*) -- the dimension along which the reduction should be done. If None, reduce all dimensions

        -   **keep_dims** (*bool*) -- if true, keep the reduced dimensions with length 1

        -   **dtype** (*tl.dtype*) -- the desired data type of the returned tensor. If specified, the input tensor is casted to `dtype` before the operation is performed. This is useful for preventing data overflows. If not specified, integer and bool dtypes are upcasted to `tl.int32` and float dtypes are upcasted to at least `tl.float32`.

    This function can also be called as a member function on `tensor`, as `x.sum(...)` instead of `sum(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

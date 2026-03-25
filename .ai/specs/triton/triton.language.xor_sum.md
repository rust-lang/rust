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
        -   triton.language.xor_sum
            -   `xor_sum()`
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
-   triton.language.xor_sum
-   View page source

------------------------------------------------------------------------

# triton.language.xor_sum¶

triton.language.xor_sum(*input*, *axis=None*, *keep_dims=False*)¶

:   Returns the xor sum of all elements in the `input` tensor along the provided `axis`

    Parameters:

    :   -   **input** (*Tensor*) -- the input values

        -   **axis** (*int*) -- the dimension along which the reduction should be done. If None, reduce all dimensions

        -   **keep_dims** (*bool*) -- if true, keep the reduced dimensions with length 1

    This function can also be called as a member function on `tensor`, as `x.xor_sum(...)` instead of `xor_sum(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

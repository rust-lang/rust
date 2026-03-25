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
            -   `cumprod()`
        -   triton.language.cumsum
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
-   triton.language.cumprod
-   View page source

------------------------------------------------------------------------

# triton.language.cumprod¶

triton.language.cumprod(*input*, *axis=0*, *reverse=False*)¶

:   Returns the cumprod of all elements in the `input` tensor along the provided `axis`

    Parameters:

    :   -   **input** (*Tensor*) -- the input values

        -   **axis** (*int*) -- the dimension along which the scan should be done

        -   **reverse** (*bool*) -- if true, the scan is performed in the reverse direction

    This function can also be called as a member function on `tensor`, as `x.cumprod(...)` instead of `cumprod(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

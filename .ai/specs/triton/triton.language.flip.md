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
        -   triton.language.flip
            -   `flip()`
        -   triton.language.where
        -   triton.language.swizzle2d
    -   Math Ops
    -   Reduction Ops
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
-   triton.language.flip
-   View page source

------------------------------------------------------------------------

# triton.language.flip¶

triton.language.flip(*x*, *dim=None*)¶

:   Flips a tensor x along the dimension dim.

    Parameters:

    :   -   **x** (*Block*) -- the first input tensor

        -   **dim** (*int*) -- the dimension to flip along

    This function can also be called as a member function on `tensor`, as `x.flip(...)` instead of `flip(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

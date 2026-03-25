Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
    -   Programming Model
    -   Creation Ops
        -   triton.language.arange
            -   `arange()`
        -   triton.language.cat
        -   triton.language.full
        -   triton.language.zeros
        -   triton.language.zeros_like
        -   triton.language.cast
    -   Shape Manipulation Ops
    -   Linear Algebra Ops
    -   Memory/Pointer Ops
    -   Indexing Ops
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
-   triton.language.arange
-   View page source

------------------------------------------------------------------------

# triton.language.arange¶

triton.language.arange(*start*, *end*, *\_semantic=None*)¶

:   Returns contiguous values within the half-open interval `[start,`` ``end)`. `end`` ``-`` ``start` must be less than or equal to `TRITON_MAX_TENSOR_NUMEL`` ``=`` ``1048576`

    Parameters:

    :   -   **start** (*int32*) -- Start of the interval. Must be a power of two.

        -   **end** (*int32*) -- End of the interval. Must be a power of two greater than `start`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

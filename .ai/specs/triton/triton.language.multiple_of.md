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
    -   Atomic Ops
    -   Random Number Generation
    -   Iterators
    -   Inline Assembly
    -   Compiler Hint Ops
        -   triton.language.assume
        -   triton.language.debug_barrier
        -   triton.language.max_constancy
        -   triton.language.max_contiguous
        -   triton.language.multiple_of
            -   `multiple_of()`
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
-   triton.language.multiple_of
-   View page source

------------------------------------------------------------------------

# triton.language.multiple_of¶

triton.language.multiple_of(*input*, *values*, *\_semantic=None*)¶

:   Let the compiler know that the values in `input` are all multiples of `value`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

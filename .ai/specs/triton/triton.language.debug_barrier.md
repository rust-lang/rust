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
            -   `debug_barrier()`
        -   triton.language.max_constancy
        -   triton.language.max_contiguous
        -   triton.language.multiple_of
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
-   triton.language.debug_barrier
-   View page source

------------------------------------------------------------------------

# triton.language.debug_barrier¶

triton.language.debug_barrier(*\_semantic=None*)¶

:   Insert a barrier to synchronize all threads in a block.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

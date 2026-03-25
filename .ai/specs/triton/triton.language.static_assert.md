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
    -   Debug Ops
        -   triton.language.static_print
        -   triton.language.static_assert
            -   `static_assert()`
        -   triton.language.device_print
        -   triton.language.device_assert
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
-   triton.language.static_assert
-   View page source

------------------------------------------------------------------------

# triton.language.static_assert¶

triton.language.static_assert(*cond*, *msg=\'\'*, *\_semantic=None*)¶

:   Assert the condition at compile time. Does not require that the `TRITON_DEBUG` environment variable is set.

        tl.static_assert(BLOCK_SIZE == 1024)

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

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
        -   triton.language.randint4x
        -   triton.language.randint
        -   triton.language.rand
        -   triton.language.randn
            -   `randn()`
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
-   triton.language.randn
-   View page source

------------------------------------------------------------------------

# triton.language.randn¶

triton.language.randn(*seed*, *offset*, *n_rounds: constexpr = constexpr\[10\]*)¶

:   Given a `seed` scalar and an `offset` block, returns a block of random `float32` in \\(\\mathcal{N}(0, 1)\\).

    Parameters:

    :   -   **seed** -- The seed for generating random numbers.

        -   **offsets** -- The offsets to generate random numbers for.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

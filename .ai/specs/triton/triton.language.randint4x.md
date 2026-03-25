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
            -   `randint4x()`
        -   triton.language.randint
        -   triton.language.rand
        -   triton.language.randn
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
-   triton.language.randint4x
-   View page source

------------------------------------------------------------------------

# triton.language.randint4x¶

triton.language.randint4x(*seed*, *offset*, *n_rounds: constexpr = constexpr\[10\]*)¶

:   Given a `seed` scalar and an `offset` block, returns four blocks of random `int32`.

    This is the maximally efficient entry point to Triton's Philox pseudo-random number generator.

    Parameters:

    :   -   **seed** -- The seed for generating random numbers.

        -   **offsets** -- The offsets to generate random numbers for.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

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
        -   triton.language.cat
        -   triton.language.full
        -   triton.language.zeros
            -   `zeros()`
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
-   triton.language.zeros
-   View page source

------------------------------------------------------------------------

# triton.language.zeros¶

triton.language.zeros(*shape*, *dtype*)¶

:   Returns a tensor filled with the scalar value 0 for the given `shape` and `dtype`.

    Parameters:

    :   -   **shape** (*tuple* *of* *ints*) -- Shape of the new array, e.g., (8, 16) or (8, )

        -   **dtype** (*DType*) -- Data-type of the new array, e.g., `tl.float16`

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

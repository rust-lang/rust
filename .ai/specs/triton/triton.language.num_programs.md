Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
    -   Programming Model
        -   triton.language.tensor
        -   triton.language.tensor_descriptor
        -   triton.language.program_id
        -   triton.language.num_programs
            -   `num_programs()`
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
-   triton.language.num_programs
-   View page source

------------------------------------------------------------------------

# triton.language.num_programs¶

triton.language.num_programs(*axis*, *\_semantic=None*)¶

:   Returns the number of program instances launched along the given `axis`.

    Parameters:

    :   **axis** (*int*) -- The axis of the 3D launch grid. Must be 0, 1 or 2.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

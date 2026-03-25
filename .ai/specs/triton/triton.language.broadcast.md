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
        -   triton.language.broadcast
            -   `broadcast()`
        -   triton.language.broadcast_to
        -   triton.language.expand_dims
        -   triton.language.interleave
        -   triton.language.join
        -   triton.language.permute
        -   triton.language.ravel
        -   triton.language.reshape
        -   triton.language.split
        -   triton.language.trans
        -   triton.language.view
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
-   triton.language.broadcast
-   View page source

------------------------------------------------------------------------

# triton.language.broadcast¶

triton.language.broadcast(*input*, *other*, *\_semantic=None*)¶

:   Tries to broadcast the two given blocks to a common compatible shape.

    Parameters:

    :   -   **input** (*Block*) -- The first input tensor.

        -   **other** (*Block*) -- The second input tensor.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

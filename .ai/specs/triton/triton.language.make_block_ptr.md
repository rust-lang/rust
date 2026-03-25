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
        -   triton.language.load
        -   triton.language.store
        -   triton.language.make_tensor_descriptor
        -   triton.language.load_tensor_descriptor
        -   triton.language.store_tensor_descriptor
        -   triton.language.make_block_ptr
            -   `make_block_ptr()`
        -   triton.language.advance
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
-   triton.language.make_block_ptr
-   View page source

------------------------------------------------------------------------

# triton.language.make_block_ptr¶

triton.language.make_block_ptr(*base: tensor*, *shape*, *strides*, *offsets*, *block_shape*, *order*, *\_semantic=None*)¶

:   Returns a pointer to a block in a parent tensor

    Parameters:

    :   -   **base** -- The base pointer to the parent tensor

        -   **shape** -- The shape of the parent tensor

        -   **strides** -- The strides of the parent tensor

        -   **offsets** -- The offsets to the block

        -   **block_shape** -- The shape of the block

        -   **order** -- The order of the original data format

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

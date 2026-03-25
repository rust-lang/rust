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
        -   triton.language.advance
            -   `advance()`
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
-   triton.language.advance
-   View page source

------------------------------------------------------------------------

# triton.language.advance¶

triton.language.advance(*base*, *offsets*, *\_semantic=None*)¶

:   Advance a block pointer

    Parameters:

    :   -   **base** -- the block pointer to advance

        -   **offsets** -- the offsets to advance, a tuple by dimension

    This function can also be called as a member function on `tensor`, as `x.advance(...)` instead of `advance(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

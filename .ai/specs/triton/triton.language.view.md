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
            -   `view()`
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
-   triton.language.view
-   View page source

------------------------------------------------------------------------

# triton.language.view¶

triton.language.view(*input*, *\*shape*, *\_semantic=None*)¶

:   Returns a tensor with the same elements as input but a different shape. The order of the elements may not be preserved.

    Parameters:

    :   -   **input** (*Block*) -- The input tensor.

        -   **shape** -- The desired shape.

    `shape` can be passed as a tuple or as individual parameters:

        # These are equivalent
        view(x, (32, 32))
        view(x, 32, 32)

    This function can also be called as a member function on `tensor`, as `x.view(...)` instead of `view(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

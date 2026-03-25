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
            -   `permute()`
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
-   triton.language.permute
-   View page source

------------------------------------------------------------------------

# triton.language.permute¶

triton.language.permute(*input*, *\*dims*, *\_semantic=None*)¶

:   Permutes the dimensions of a tensor.

    Parameters:

    :   -   **input** (*Block*) -- The input tensor.

        -   **dims** -- The desired ordering of dimensions. For example, `(2,`` ``1,`` ``0)` reverses the order dims in a 3D tensor.

    `dims` can be passed as a tuple or as individual parameters:

        # These are equivalent
        permute(x, (2, 1, 0))
        permute(x, 2, 1, 0)

    `trans()` is equivalent to this function, except when `dims` is empty, it tries to swap the last two axes.

    This function can also be called as a member function on `tensor`, as `x.permute(...)` instead of `permute(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

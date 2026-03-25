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
            -   `trans()`
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
-   triton.language.trans
-   View page source

------------------------------------------------------------------------

# triton.language.trans¶

triton.language.trans(*input: tensor*, *\*dims*, *\_semantic=None*)¶

:   Permutes the dimensions of a tensor.

    If the parameter `dims` is not specified, the function defaults to swapping the last two axes, thereby performing an (optionally batched) 2D transpose.

    Parameters:

    :   -   **input** -- The input tensor.

        -   **dims** -- The desired ordering of dimensions. For example, `(2,`` ``1,`` ``0)` reverses the order dims in a 3D tensor.

    `dims` can be passed as a tuple or as individual parameters:

        # These are equivalent
        trans(x, (2, 1, 0))
        trans(x, 2, 1, 0)

    `permute()` is equivalent to this function, except it doesn't have the special case when no permutation is specified.

    This function can also be called as a member function on `tensor`, as `x.trans(...)` instead of `trans(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

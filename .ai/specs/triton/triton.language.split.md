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
            -   `split()`
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
-   triton.language.split
-   View page source

------------------------------------------------------------------------

# triton.language.split¶

triton.language.split(*a*, *\_semantic=None*, *\_generator=None*) → tuple\[tensor, tensor\]¶

:   Split a tensor in two along its last dim, which must have size 2.

    For example, given a tensor of shape (4,8,2), produces two tensors of shape (4,8). Given a tensor of shape (2), returns two scalars.

    If you want to split into more than two pieces, you can use multiple calls to this function (probably plus calling reshape). This reflects the constraint in Triton that tensors must have power-of-two sizes.

    split is the inverse of join.

    Parameters:

    :   **a** (*Tensor*) -- The tensor to split.

    This function can also be called as a member function on `tensor`, as `x.split()` instead of `split(x)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

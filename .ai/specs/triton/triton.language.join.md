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
            -   `join()`
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
-   triton.language.join
-   View page source

------------------------------------------------------------------------

# triton.language.join¶

triton.language.join(*a*, *b*, *\_semantic=None*)¶

:   Join the given tensors in a new, minor dimension.

    For example, given two tensors of shape (4,8), produces a new tensor of shape (4,8,2). Given two scalars, returns a tensor of shape (2).

    The two inputs are broadcasted to be the same shape.

    If you want to join more than two elements, you can use multiple calls to this function. This reflects the constraint in Triton that tensors must have power-of-two sizes.

    join is the inverse of split.

    Parameters:

    :   -   **a** (*Tensor*) -- The first input tensor.

        -   **b** (*Tensor*) -- The second input tensor.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

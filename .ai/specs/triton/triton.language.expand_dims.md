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
            -   `expand_dims()`
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
-   triton.language.expand_dims
-   View page source

------------------------------------------------------------------------

# triton.language.expand_dims¶

triton.language.expand_dims(*input*, *axis*, *\_semantic=None*)¶

:   Expand the shape of a tensor, by inserting new length-1 dimensions.

    Axis indices are with respect to the resulting tensor, so `result.shape[axis]` will be 1 for each axis.

    Parameters:

    :   -   **input** (*tl.tensor*) -- The input tensor.

        -   **axis** (*int* *\|* *Sequence\[int\]*) -- The indices to add new axes

    This function can also be called as a member function on `tensor`, as `x.expand_dims(...)` instead of `expand_dims(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

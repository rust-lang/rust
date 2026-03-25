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
    -   Indexing Ops
    -   Math Ops
    -   Reduction Ops
        -   triton.language.argmax
        -   triton.language.argmin
        -   triton.language.max
        -   triton.language.min
        -   triton.language.reduce
            -   `reduce()`
        -   triton.language.sum
        -   triton.language.xor_sum
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
-   triton.language.reduce
-   View page source

------------------------------------------------------------------------

# triton.language.reduce¶

triton.language.reduce(*input*, *axis*, *combine_fn*, *keep_dims=False*, *\_semantic=None*, *\_generator=None*)¶

:   Applies the combine_fn to all elements in `input` tensors along the provided `axis`

    Parameters:

    :   -   **input** (*Tensor*) -- the input tensor, or tuple of tensors

        -   **axis** (*int* *\|* *None*) -- the dimension along which the reduction should be done. If None, reduce all dimensions

        -   **combine_fn** (*Callable*) -- a function to combine two groups of scalar tensors (must be marked with \@triton.jit)

        -   **keep_dims** (*bool*) -- if true, keep the reduced dimensions with length 1

    This function can also be called as a member function on `tensor`, as `x.reduce(...)` instead of `reduce(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

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
    -   Scan/Sort Ops
        -   triton.language.associative_scan
            -   `associative_scan()`
        -   triton.language.cumprod
        -   triton.language.cumsum
        -   triton.language.histogram
        -   triton.language.sort
        -   triton.language.gather
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
-   triton.language.associative_scan
-   View page source

------------------------------------------------------------------------

# triton.language.associative_scan¶

triton.language.associative_scan(*input*, *axis*, *combine_fn*, *reverse=False*, *\_semantic=None*, *\_generator=None*)¶

:   Applies the combine_fn to each elements with a carry in `input` tensors along the provided `axis` and update the carry

    Parameters:

    :   -   **input** (*Tensor*) -- the input tensor, or tuple of tensors

        -   **axis** (*int*) -- the dimension along which the reduction should be done

        -   **combine_fn** (*Callable*) -- a function to combine two groups of scalar tensors (must be marked with \@triton.jit)

        -   **reverse** (*bool*) -- whether to apply the associative scan in the reverse direction along axis

    This function can also be called as a member function on `tensor`, as `x.associative_scan(...)` instead of `associative_scan(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

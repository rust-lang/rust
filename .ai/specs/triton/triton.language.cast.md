Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
    -   Programming Model
    -   Creation Ops
        -   triton.language.arange
        -   triton.language.cat
        -   triton.language.full
        -   triton.language.zeros
        -   triton.language.zeros_like
        -   triton.language.cast
            -   `cast()`
    -   Shape Manipulation Ops
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
-   triton.language.cast
-   View page source

------------------------------------------------------------------------

# triton.language.cast¶

triton.language.cast(*input*, *dtype: dtype*, *fp_downcast_rounding: str \| None = None*, *bitcast: bool = False*, *\_semantic=None*)¶

:   Casts a tensor to the given `dtype`.

    Parameters:

    :   -   **dtype** (*tl.dtype*) -- The target data type.

        -   **fp_downcast_rounding** (*str,* *optional*) -- The rounding mode for downcasting floating-point values. This parameter is only used when self is a floating-point tensor and dtype is a floating-point type with a smaller bitwidth. Supported values are `"rtne"` (round to nearest, ties to even) and `"rtz"` (round towards zero).

        -   **bitcast** (*bool,* *optional*) -- If true, the tensor is bitcasted to the given `dtype`, instead of being numerically casted.

    This function can also be called as a member function on `tensor`, as `x.cast(...)` instead of `cast(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

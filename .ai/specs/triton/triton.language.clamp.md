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
        -   triton.language.abs
        -   triton.language.cdiv
        -   triton.language.ceil
        -   triton.language.clamp
            -   `clamp()`
        -   triton.language.cos
        -   triton.language.div_rn
        -   triton.language.erf
        -   triton.language.exp
        -   triton.language.exp2
        -   triton.language.fdiv
        -   triton.language.floor
        -   triton.language.fma
        -   triton.language.log
        -   triton.language.log2
        -   triton.language.maximum
        -   triton.language.minimum
        -   triton.language.rsqrt
        -   triton.language.sigmoid
        -   triton.language.sin
        -   triton.language.softmax
        -   triton.language.sqrt
        -   triton.language.sqrt_rn
        -   triton.language.umulhi
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
-   triton.language.clamp
-   View page source

------------------------------------------------------------------------

# triton.language.clamp¶

triton.language.clamp(*x*, *min*, *max*, *propagate_nan: \~triton.language.core.constexpr = \<PROPAGATE_NAN.NONE: 0\>*, *\_semantic=None*)¶

:   Clamps the input tensor `x` within the range \[min, max\]. Behavior when `min` \> `max` is undefined.

    Parameters:

    :   -   **x** (*Block*) -- the input tensor

        -   **min** (*Block*) -- the lower bound for clamping

        -   **max** (*Block*) -- the upper bound for clamping

        -   **propagate_nan** (*tl.PropagateNan*) -- whether to propagate NaN values. Applies only to the `x` tensor. If either `min` or `max` is NaN, the result is undefined.

    See also

    `tl.PropagateNan`

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

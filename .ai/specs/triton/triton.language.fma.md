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
        -   triton.language.cos
        -   triton.language.div_rn
        -   triton.language.erf
        -   triton.language.exp
        -   triton.language.exp2
        -   triton.language.fdiv
        -   triton.language.floor
        -   triton.language.fma
            -   `fma()`
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
-   triton.language.fma
-   View page source

------------------------------------------------------------------------

# triton.language.fma¶

triton.language.fma(*x*, *y*, *z*, *\_semantic=None*)¶

:   Computes the element-wise fused multiply-add of `x`, `y`, and `z`.

    Parameters:

    :   -   **x** (*Block*) -- the input values

        -   **y** (*Block*) -- the input values

        -   **z** (*Block*) -- the input values

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

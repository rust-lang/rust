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
    -   Atomic Ops
    -   Random Number Generation
    -   Iterators
        -   triton.language.range
        -   triton.language.static_range
            -   `static_range`
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
-   triton.language.static_range
-   View page source

------------------------------------------------------------------------

# triton.language.static_range¶

*class *triton.language.static_range(*self*, *arg1*, *arg2=None*, *step=None*)¶

:   Iterator that counts upward forever.

        @triton.jit
        def kernel(...):
            for i in tl.static_range(10):
                ...

    Note:

    :   This is a special iterator used to implement similar semantics to Python's `range` in the context of `triton.jit` functions. In addition, it also guides the compiler to unroll the loop aggressively.

    Parameters:

    :   -   **arg1** -- the start value.

        -   **arg2** -- the end value.

        -   **step** -- the step value.

    \_\_init\_\_(*self*, *arg1*, *arg2=None*, *step=None*)¶

    :   

    Methods

      ---------------------------------------- --
      `__init__`(self, arg1\[, arg2, step\])   
      ---------------------------------------- --

    Attributes

      -------- --
      `type`   
      -------- --

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

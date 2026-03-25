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
    -   Inline Assembly
    -   Compiler Hint Ops
    -   Debug Ops
        -   triton.language.static_print
        -   triton.language.static_assert
        -   triton.language.device_print
        -   triton.language.device_assert
            -   `device_assert()`
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
-   triton.language.device_assert
-   View page source

------------------------------------------------------------------------

# triton.language.device_assert¶

triton.language.device_assert(*cond*, *msg=\'\'*, *mask=None*, *\_semantic=None*)¶

:   Assert the condition at runtime from the device. Requires that the environment variable `TRITON_DEBUG` is set to a value besides `0` in order for this to have any effect.

    Using the Python `assert` statement is the same as calling this function, except that the second argument must be provided and must be a string, e.g. `assert`` ``pid`` ``==`` ``0,`` ``"pid`` ``!=`` ``0"`. The environment variable must be set for this `assert` statement to have any effect.

        tl.device_assert(pid == 0)
        assert pid == 0, f"pid != 0"

    Parameters:

    :   -   **cond** -- the condition to assert. This is required to be a boolean tensor.

        -   **msg** -- the message to print if the assertion fails. This is required to be a string literal.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

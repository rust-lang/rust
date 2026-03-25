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
            -   `static_print()`
        -   triton.language.static_assert
        -   triton.language.device_print
        -   triton.language.device_assert
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
-   triton.language.static_print
-   View page source

------------------------------------------------------------------------

# triton.language.static_print¶

triton.language.static_print(*\*values*, *sep: str = \' \'*, *end: str = \'\\n\'*, *file=None*, *flush=False*, *\_semantic=None*)¶

:   Print the values at compile time. The parameters are the same as the builtin `print`.

    NOTE: Calling the Python builtin `print` is not the same as calling this, it instead maps to `device_print`, which has special requirements for the arguments.

        tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

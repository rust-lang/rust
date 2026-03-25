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
            -   `device_print()`
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
-   triton.language.device_print
-   View page source

------------------------------------------------------------------------

# triton.language.device_print¶

triton.language.device_print(*prefix*, *\*args*, *hex=False*, *\_semantic=None*)¶

:   Print the values at runtime from the device. String formatting does not work for runtime values, so you should provide the values you want to print as arguments. The first value must be a string, all following values must be scalars or tensors.

    Calling the Python builtin `print` is the same as calling this function, and the requirements for the arguments will match this function (not the normal requirements for `print`).

        tl.device_print("pid", pid)
        print("pid", pid)

    On CUDA, printfs are streamed through a buffer of limited size (on one host, we measured the default as 6912 KiB, but this may not be consistent across GPUs and CUDA versions). If you notice some printfs are being dropped, you can increase the buffer size by calling

        triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)

    CUDA may raise an error if you try to change this value after running a kernel that uses printfs. The value set here may only affect the current device (so if you have multiple GPUs, you'd need to call it multiple times).

    Parameters:

    :   -   **prefix** -- a prefix to print before the values. This is required to be a string literal.

        -   **args** -- the values to print. They can be any tensor or scalar.

        -   **hex** -- print all values as hex instead of decimal

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

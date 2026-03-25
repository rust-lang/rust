Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
    -   triton.jit
        -   `jit()`
    -   triton.autotune
    -   triton.heuristics
    -   triton.Config
-   triton.language
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
-   triton
-   triton.jit
-   View page source

------------------------------------------------------------------------

# triton.jit¶

triton.jit(*fn: T*) → JITFunction\[T\]¶\
triton.jit(*\**, *version=None*, *repr: Callable \| None = None*, *launch_metadata: Callable \| None = None*, *do_not_specialize: Iterable\[int \| str\] \| None = None*, *do_not_specialize_on_alignment: Iterable\[int \| str\] \| None = None*, *debug: bool \| None = None*, *noinline: bool \| None = None*) → Callable\[\[T\], JITFunction\[T\]\]

:   Decorator for JIT-compiling a function using the Triton compiler.

    Note:

    :   When a jit'd function is called, arguments are implicitly converted to pointers if they have a `.data_ptr()` method and a .dtype attribute.

    Note:

    :   This function will be compiled and run on the GPU. It will only have access to:

        -   python primitives,

        -   builtins within the triton package,

        -   arguments to this function,

        -   other jit'd functions

    Parameters:

    :   **fn** (*Callable*) -- the function to be jit-compiled

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.

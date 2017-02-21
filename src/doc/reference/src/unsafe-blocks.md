# Unsafe blocks

A block of code can be prefixed with the `unsafe` keyword, to permit calling
`unsafe` functions or dereferencing raw pointers within a safe function.

When a programmer has sufficient conviction that a sequence of potentially
unsafe operations is actually safe, they can encapsulate that sequence (taken
as a whole) within an `unsafe` block. The compiler will consider uses of such
code safe, in the surrounding context.

Unsafe blocks are used to wrap foreign libraries, make direct use of hardware
or implement features not directly present in the language. For example, Rust
provides the language features necessary to implement memory-safe concurrency
in the language but the implementation of threads and message passing is in the
standard library.

Rust's type system is a conservative approximation of the dynamic safety
requirements, so in some cases there is a performance cost to using safe code.
For example, a doubly-linked list is not a tree structure and can only be
represented with reference-counted pointers in safe code. By using `unsafe`
blocks to represent the reverse links as raw pointers, it can be implemented
with only boxes.

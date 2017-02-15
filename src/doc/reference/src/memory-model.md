# Memory model

A Rust program's memory consists of a static set of *items* and a *heap*.
Immutable portions of the heap may be safely shared between threads, mutable
portions may not be safely shared, but several mechanisms for effectively-safe
sharing of mutable values, built on unsafe code but enforcing a safe locking
discipline, exist in the standard library.

Allocations in the stack consist of *variables*, and allocations in the heap
consist of *boxes*.

## Memory allocation and lifetime

The _items_ of a program are those functions, modules and types that have their
value calculated at compile-time and stored uniquely in the memory image of the
rust process. Items are neither dynamically allocated nor freed.

The _heap_ is a general term that describes boxes.  The lifetime of an
allocation in the heap depends on the lifetime of the box values pointing to
it. Since box values may themselves be passed in and out of frames, or stored
in the heap, heap allocations may outlive the frame they are allocated within.
An allocation in the heap is guaranteed to reside at a single location in the
heap for the whole lifetime of the allocation - it will never be relocated as
a result of moving a box value.

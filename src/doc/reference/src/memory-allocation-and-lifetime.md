# Memory allocation and lifetime

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

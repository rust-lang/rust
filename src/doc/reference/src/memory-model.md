# Memory model

A Rust program's memory consists of a static set of *items* and a *heap*.
Immutable portions of the heap may be safely shared between threads, mutable
portions may not be safely shared, but several mechanisms for effectively-safe
sharing of mutable values, built on unsafe code but enforcing a safe locking
discipline, exist in the standard library.

Allocations in the stack consist of *variables*, and allocations in the heap
consist of *boxes*.

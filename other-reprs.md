% Alternative representations

Rust allows you to specify alternative data layout strategies from the default.




# repr(C)

This is the most important `repr`. It has fairly simple intent: do what C does.
The order, size, and alignment of fields is exactly what you would expect from
C or C++. Any type you expect to pass through an FFI boundary should have `repr(C)`,
as C is the lingua-franca of the programming world. This is also necessary
to soundly do more elaborate tricks with data layout such as reintepretting values
as a different type.

However, the interaction with Rust's more exotic data layout features must be kept
in mind. Due to its dual purpose as "for FFI" and "for layout control", `repr(C)`
can be applied to types that will be nonsensical or problematic if passed through
the FFI boundary.

* ZSTs are still zero-sized, even though this is not a standard behaviour
  in C, and is explicitly contrary to the behaviour of an empty type in C++, which
  still consumes a byte of space.

* DSTs, tuples, and tagged unions are not a concept in C and as such are never
  FFI safe.

* **The drop flag will still be added**

* This is equivalent to `repr(u32)` for enums (see below)




# repr(packed)

`repr(packed)` forces rust to strip any padding, and only align the type to a
byte. This may improve the memory footprint, but will likely have other
negative side-effects.

In particular, most architectures *strongly* prefer values to be aligned. This
may mean the unaligned loads are penalized (x86), or even fault (ARM). In
particular, the compiler may have trouble with references to unaligned fields.

`repr(packed)` is not to be used lightly. Unless you have extreme requirements,
this should not be used.

This repr is a modifier on `repr(C)` and `repr(rust)`.




# repr(u8), repr(u16), repr(u32), repr(u64)

These specify the size to make a C-like enum. If the discriminant overflows the
integer it has to fit in, it will be an error. You can manually ask Rust to
allow this by setting the overflowing element to explicitly be 0. However Rust
will not allow you to create an enum where two variants.

These reprs have no affect on a struct or non-C-like enum.
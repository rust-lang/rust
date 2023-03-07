A `#[simd]` attribute was applied to an empty tuple struct.

Erroneous code example:

```compile_fail,E0075
#![feature(repr_simd)]

#[repr(simd)]
struct Bad; // error!
```

The `#[simd]` attribute can only be applied to non empty tuple structs, because
it doesn't make sense to try to use SIMD operations when there are no values to
operate on.

Fixed example:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32); // ok!
```

All types in a tuple struct aren't the same when using the `#[simd]`
attribute.

Erroneous code example:

```compile_fail,E0076
#![feature(repr_simd)]

#[repr(simd)]
struct Bad(u16, u32, u32 u32); // error!
```

When using the `#[simd]` attribute to automatically use SIMD operations in tuple
struct, the types in the struct must all be of the same type, or the compiler
will trigger this error.

Fixed example:

```
#![feature(repr_simd)]

#[repr(simd)]
struct Good(u32, u32, u32, u32); // ok!
```

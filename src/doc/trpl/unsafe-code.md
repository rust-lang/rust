% Raw Pointers

Rust offers two additional pointer types (*raw pointers*), written as
`*const T` and `*mut T`. They're an approximation of C's `const T*` and `T*`
respectively; indeed, one of their most common uses is for FFI,
interfacing with external C libraries.

Raw pointers have much fewer guarantees than other pointer types
offered by the Rust language and libraries. For example, they

- are not guaranteed to point to valid memory and are not even
  guaranteed to be non-null (unlike both `Box` and `&`);
- do not have any automatic clean-up, unlike `Box`, and so require
  manual resource management;
- are plain-old-data, that is, they don't move ownership, again unlike
  `Box`, hence the Rust compiler cannot protect against bugs like
  use-after-free;
- lack any form of lifetimes, unlike `&`, and so the compiler cannot
  reason about dangling pointers; and
- have no guarantees about aliasing or mutability other than mutation
  not being allowed directly through a `*const T`.

Fortunately, they come with a redeeming feature: the weaker guarantees
mean weaker restrictions. The missing restrictions make raw pointers
appropriate as a building block for implementing things like smart
pointers and vectors inside libraries. For example, `*` pointers are
allowed to alias, allowing them to be used to write shared-ownership
types like reference counted and garbage collected pointers, and even
thread-safe shared memory types (`Rc` and the `Arc` types are both
implemented entirely in Rust).

There are two things that you are required to be careful about
(i.e. require an `unsafe { ... }` block) with raw pointers:

- dereferencing: they can have any value: so possible results include
  a crash, a read of uninitialised memory, a use-after-free, or
  reading data as normal.
- pointer arithmetic via the `offset` [intrinsic](#intrinsics) (or
  `.offset` method): this intrinsic uses so-called "in-bounds"
  arithmetic, that is, it is only defined behaviour if the result is
  inside (or one-byte-past-the-end) of the object from which the
  original pointer came.

The latter assumption allows the compiler to optimize more
effectively. As can be seen, actually *creating* a raw pointer is not
unsafe, and neither is converting to an integer.

### References and raw pointers

At runtime, a raw pointer `*` and a reference pointing to the same
piece of data have an identical representation. In fact, an `&T`
reference will implicitly coerce to an `*const T` raw pointer in safe code
and similarly for the `mut` variants (both coercions can be performed
explicitly with, respectively, `value as *const T` and `value as *mut T`).

Going the opposite direction, from `*const` to a reference `&`, is not
safe. A `&T` is always valid, and so, at a minimum, the raw pointer
`*const T` has to point to a valid instance of type `T`. Furthermore,
the resulting pointer must satisfy the aliasing and mutability laws of
references. The compiler assumes these properties are true for any
references, no matter how they are created, and so any conversion from
raw pointers is asserting that they hold. The programmer *must*
guarantee this.

The recommended method for the conversion is

```
let i: u32 = 1;
// explicit cast
let p_imm: *const u32 = &i as *const u32;
let mut m: u32 = 2;
// implicit coercion
let p_mut: *mut u32 = &mut m;

unsafe {
    let ref_imm: &u32 = &*p_imm;
    let ref_mut: &mut u32 = &mut *p_mut;
}
```

The `&*x` dereferencing style is preferred to using a `transmute`.
The latter is far more powerful than necessary, and the more
restricted operation is harder to use incorrectly; for example, it
requires that `x` is a pointer (unlike `transmute`).

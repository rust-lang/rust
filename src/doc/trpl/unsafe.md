% Unsafe and Low-Level Code

# Introduction

Rust aims to provide safe abstractions over the low-level details of
the CPU and operating system, but sometimes one needs to drop down and
write code at that level. This guide aims to provide an overview of
the dangers and power one gets with Rust's unsafe subset.

Rust provides an escape hatch in the form of the `unsafe { ... }`
block which allows the programmer to dodge some of the compiler's
checks and do a wide range of operations, such as:

- dereferencing [raw pointers](#raw-pointers)
- calling a function via FFI ([covered by the FFI guide](ffi.html))
- casting between types bitwise (`transmute`, aka "reinterpret cast")
- [inline assembly](#inline-assembly)

Note that an `unsafe` block does not relax the rules about lifetimes
of `&` and the freezing of borrowed data.

Any use of `unsafe` is the programmer saying "I know more than you" to
the compiler, and, as such, the programmer should be very sure that
they actually do know more about why that piece of code is valid.  In
general, one should try to minimize the amount of unsafe code in a
code base; preferably by using the bare minimum `unsafe` blocks to
build safe interfaces.

> **Note**: the low-level details of the Rust language are still in
> flux, and there is no guarantee of stability or backwards
> compatibility. In particular, there may be changes that do not cause
> compilation errors, but do cause semantic changes (such as invoking
> undefined behaviour). As such, extreme care is required.

# Pointers

## References

One of Rust's biggest features is memory safety.  This is achieved in
part via [the ownership system](ownership.html), which is how the
compiler can guarantee that every `&` reference is always valid, and,
for example, never pointing to freed memory.

These restrictions on `&` have huge advantages. However, they also
constrain how we can use them. For example, `&` doesn't behave
identically to C's pointers, and so cannot be used for pointers in
foreign function interfaces (FFI). Additionally, both immutable (`&`)
and mutable (`&mut`) references have some aliasing and freezing
guarantees, required for memory safety.

In particular, if you have an `&T` reference, then the `T` must not be
modified through that reference or any other reference. There are some
standard library types, e.g. `Cell` and `RefCell`, that provide inner
mutability by replacing compile time guarantees with dynamic checks at
runtime.

An `&mut` reference has a different constraint: when an object has an
`&mut T` pointing into it, then that `&mut` reference must be the only
such usable path to that object in the whole program. That is, an
`&mut` cannot alias with any other references.

Using `unsafe` code to incorrectly circumvent and violate these
restrictions is undefined behaviour. For example, the following
creates two aliasing `&mut` pointers, and is invalid.

```
use std::mem;
let mut x: u8 = 1;

let ref_1: &mut u8 = &mut x;
let ref_2: &mut u8 = unsafe { mem::transmute(&mut *ref_1) };

// oops, ref_1 and ref_2 point to the same piece of data (x) and are
// both usable
*ref_1 = 10;
*ref_2 = 20;
```

## Raw pointers

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



## Making the unsafe safe(r)

There are various ways to expose a safe interface around some unsafe
code:

- store pointers privately (i.e. not in public fields of public
  structs), so that you can see and control all reads and writes to
  the pointer in one place.
- use `assert!()` a lot: since you can't rely on the protection of the
  compiler & type-system to ensure that your `unsafe` code is correct
  at compile-time, use `assert!()` to verify that it is doing the
  right thing at run-time.
- implement the `Drop` for resource clean-up via a destructor, and use
  RAII (Resource Acquisition Is Initialization). This reduces the need
  for any manual memory management by users, and automatically ensures
  that clean-up is always run, even when the thread panics.
- ensure that any data stored behind a raw pointer is destroyed at the
  appropriate time.

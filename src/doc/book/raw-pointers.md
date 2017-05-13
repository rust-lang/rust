% Raw Pointers

Rust has a number of different smart pointer types in its standard library, but
there are two types that are extra-special. Much of Rust’s safety comes from
compile-time checks, but raw pointers don’t have such guarantees, and are
[unsafe][unsafe] to use.

`*const T` and `*mut T` are called ‘raw pointers’ in Rust. Sometimes, when
writing certain kinds of libraries, you’ll need to get around Rust’s safety
guarantees for some reason. In this case, you can use raw pointers to implement
your library, while exposing a safe interface for your users. For example, `*`
pointers are allowed to alias, allowing them to be used to write
shared-ownership types, and even thread-safe shared memory types (the `Rc<T>`
and `Arc<T>` types are both implemented entirely in Rust).

Here are some things to remember about raw pointers that are different than
other pointer types. They:

- are not guaranteed to point to valid memory and are not even
  guaranteed to be non-NULL (unlike both `Box` and `&`);
- do not have any automatic clean-up, unlike `Box`, and so require
  manual resource management;
- are plain-old-data, that is, they don't move ownership, again unlike
  `Box`, hence the Rust compiler cannot protect against bugs like
  use-after-free;
- lack any form of lifetimes, unlike `&`, and so the compiler cannot
  reason about dangling pointers; and
- have no guarantees about aliasing or mutability other than mutation
  not being allowed directly through a `*const T`.

# Basics

Creating a raw pointer is perfectly safe:

```rust
let x = 5;
let raw = &x as *const i32;

let mut y = 10;
let raw_mut = &mut y as *mut i32;
```

However, dereferencing one is not. This won’t work:

```rust,ignore
let x = 5;
let raw = &x as *const i32;

println!("raw points at {}", *raw);
```

It gives this error:

```text
error: dereference of raw pointer requires unsafe function or block [E0133]
     println!("raw points at {}", *raw);
                                  ^~~~
```

When you dereference a raw pointer, you’re taking responsibility that it’s not
pointing somewhere that would be incorrect. As such, you need `unsafe`:

```rust
let x = 5;
let raw = &x as *const i32;

let points_at = unsafe { *raw };

println!("raw points at {}", points_at);
```

For more operations on raw pointers, see [their API documentation][rawapi].

[unsafe]: unsafe.html
[rawapi]: ../std/primitive.pointer.html

# FFI

Raw pointers are useful for FFI: Rust’s `*const T` and `*mut T` are similar to
C’s `const T*` and `T*`, respectively. For more about this use, consult the
[FFI chapter][ffi].

[ffi]: ffi.html

# References and raw pointers

At runtime, a raw pointer `*` and a reference pointing to the same piece of
data have an identical representation. In fact, an `&T` reference will
implicitly coerce to an `*const T` raw pointer in safe code and similarly for
the `mut` variants (both coercions can be performed explicitly with,
respectively, `value as *const T` and `value as *mut T`).

Going the opposite direction, from `*const` to a reference `&`, is not safe. A
`&T` is always valid, and so, at a minimum, the raw pointer `*const T` has to
point to a valid instance of type `T`. Furthermore, the resulting pointer must
satisfy the aliasing and mutability laws of references. The compiler assumes
these properties are true for any references, no matter how they are created,
and so any conversion from raw pointers is asserting that they hold. The
programmer *must* guarantee this.

The recommended method for the conversion is:

```rust
// Explicit cast:
let i: u32 = 1;
let p_imm: *const u32 = &i as *const u32;

// Implicit coercion:
let mut m: u32 = 2;
let p_mut: *mut u32 = &mut m;

unsafe {
    let ref_imm: &u32 = &*p_imm;
    let ref_mut: &mut u32 = &mut *p_mut;
}
```

The `&*x` dereferencing style is preferred to using a `transmute`. The latter
is far more powerful than necessary, and the more restricted operation is
harder to use incorrectly; for example, it requires that `x` is a pointer
(unlike `transmute`).

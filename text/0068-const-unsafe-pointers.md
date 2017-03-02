- Start Date: 2014-06-11
- RFC PR: [rust-lang/rfcs#68](https://github.com/rust-lang/rfcs/pull/68)
- Rust Issue: [rust-lang/rust#7362](https://github.com/rust-lang/rust/issues/7362)

# Summary

Rename `*T` to `*const T`, retain all other semantics of unsafe pointers.

# Motivation

Currently the `T*` type in C is equivalent to `*mut T` in Rust, and the `const
T*` type in C is equivalent to the `*T` type in Rust. Noticeably, the two most
similar types, `T*` and `*T` have different meanings in Rust and C, frequently
causing confusion and often incorrect declarations of C functions.

If the compiler is ever to take advantage of the guarantees of declaring an FFI
function as taking `T*` or `const T*` (in C), then it is crucial that the FFI
declarations in Rust are faithful to the declaration in C.

The current difference in Rust unsafe pointers types with C pointers types is
proving to be too error prone to realistically enable these optimizations at a
future date. By renaming Rust's unsafe pointers to closely match their C
brethren, the likelihood for erroneously transcribing a signature is diminished.

# Detailed design

> This section will assume that the current unsafe pointer design is forgotten
> completely, and will explain the unsafe pointer design from scratch.

There are two unsafe pointers in rust, `*mut T` and `*const T`. These two types
are primarily useful when interacting with foreign functions through a FFI. The
`*mut T` type is equivalent to the `T*` type in C, and the `*const T` type is
equivalent to the `const T*` type in C.

The type `&mut T` will automatically coerce to `*mut T` in the normal locations
that coercion occurs today. It will also be possible to explicitly cast with an
`as` expression. Additionally, the `&T` type will automatically coerce to
`*const T`.  Note that `&mut T` will not automatically coerce to `*const T`.

The two unsafe pointer types will be freely castable among one another via `as`
expressions, but no coercion will occur between the two. Additionally, values of
type `uint` can be casted to unsafe pointers.

## When is a coercion valid?

When coercing from `&'a T` to `*const T`, Rust will guarantee that the memory
will remain valid for the lifetime `'a` and the memory will be immutable up to
memory stored in `Unsafe<U>`. It is the responsibility of the code working with
the `*const T` that the pointer is only dereferenced in the lifetime `'a`.

When coercing from `&'a mut T` to `*mut T`, Rust will guarantee that the memory
will stay valid during `'a` and that the memory will *not be accessed* during
`'a`. Additionally, Rust will *consume* the `&'a mut T` during the coercion. It
is the responsibility of the code working with the `*mut T` to guarantee that
the unsafe pointer is only dereferenced in the lifetime `'a`, and that the
memory is "valid again" after `'a`.

> **Note**: Rust will consume `&mut T` coercions with both implicit and explicit
> coercions.

The term "valid again" is used to represent that some types in Rust require
internal invariants, such as `Box<T>` never being `NULL`. This is often a
per-type invariant, so it is the responsibility of the unsafe code to uphold
these invariants.

## When is a safe cast valid?

Unsafe code can convert an unsafe pointer to a safe pointer via dereferencing
inside of an unsafe block. This section will discuss when this action is valid.

When converting `*mut T` to `&'a mut T`, it must be guaranteed that the memory
is initialized to start out with and that nobody will access the memory during
`'a` except for the converted pointer.

When converting `*const T` to `&'a T`, it must be guaranteed that the memory is
initialized to start out with and that nobody will write to the pointer during
`'a` except for memory within `Unsafe<U>`.

# Drawbacks

Today's unsafe pointers design is consistent with the borrowed pointers types in
Rust, using the `mut` qualifier for a mutable pointer, and no qualifier for an
"immutable" pointer. Renaming the pointers would be divergence from this
consistency, and would also introduce a keyword that is not used elsewhere in
the language, `const`.

# Alternatives

* The current `*mut T` type could be removed entirely, leaving only one unsafe
  pointer type, `*T`. This will not allow FFI calls to take advantage of the
  `const T*` optimizations on the caller side of the function. Additionally,
  this may not accurately express to the programmer what a FFI API is intending
  to do. Note, however, that other variants of unsafe pointer types could likely
  be added in the future in a backwards-compatible way.

* More effort could be invested in auto-generating bindings, and hand-generating
  bindings could be greatly discouraged. This would maintain consistency with
  Rust pointer types, and it would allow APIs to usually being transcribed
  accurately by automating the process. It is unknown how realistic this
  solution is as it is currently not yet implemented. There may still be
  confusion as well that `*T` is not equivalent to C's `T*`.

# Unresolved questions

* How much can the compiler help out when coercing `&mut T` to `*mut T`? As
  previously stated, the source pointer `&mut T` is consumed during the
  coerction (it's already a linear type), but this can lead to some unexpected
  results:

      extern {
          fn bar(a: *mut int, b: *mut int);
      }

      fn foo(a: &mut int) {
          unsafe {
              bar(&mut *a, &mut *a);
          }
      }

  This code is invalid because it is creating two copies of the same mutable
  pointer, and the external function is unaware that the two pointers alias. The
  rule that the programmer has violated is that the pointer `*mut T` is only
  dereferenced during the lifetime of the `&'a mut T` pointer. For example, here
  are the lifetimes spelled out:

      fn foo(a: &mut int) {
          unsafe {
              bar(&mut *a, &mut *a);
      //          |-----|  |-----|
      //             |        |
      //             |       Lifetime of second argument
      //            Lifetime of first argument
          }
      }

  Here it can be seen that it is impossible for the C code to safely dereference
  the pointers passed in because lifetimes don't extend into the function call
  itself. The compiler could, in this case, *extend the lifetime* of a coerced
  pointer to follow the otherwise applied temporary rules for expressions.

  In the example above, the compiler's temporary lifetime rules would cause the
  first coercion to last for the entire lifetime of the call to `bar`, thereby
  disallowing the second reborrow because it has an overlapping lifetime with
  the first.

  It is currently an open question how necessary this sort of treatment will be,
  and this lifetime treatment will likely require a new RFC.

* Will all pointer types in C need to have their own keyword in Rust for
  representation in the FFI?

* To what degree will the compiler emit metadata about FFI function calls in
  order to take advantage of optimizations on the caller side of a function
  call? Do the theoretical wins justify the scope of this redesign? There is
  currently no concrete data measuring what benefits could be gained from
  informing optimization passes about const vs non-const pointers.

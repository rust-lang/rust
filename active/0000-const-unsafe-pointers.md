- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

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
brethren, the likelihood for errneously transcribing a signature is diminished.

# Detailed design

> This section will assume that the current unsafe pointer design is forgotten
> completely, and will explain the unsafe pointer design from scratch.

There are two unsafe pointers in rust, `*mut cT` and `*const T`. These two types
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

# Drawbacks

Today's unsafe pointers design is consistent with the borrowed pointers types in
Rust, using the `mut` qualifier for a mutable pointer, and no qualifier for an
"immutable" pointer. Renaming the pointers would be divergence from this
consistency, and would also introduce a keyword that is not used elsehwere in
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

* Will all pointer types in C need to have their own keyword in Rust for
  representation in the FFI?

* To what degree will the compiler emit metadata about FFI function calls in
  order to take advantage of optimizations on the caller side of a function
  call? Do the theoretical wins justify the scope of this redesign? There is
  currently no concrete data measuring what benefits could be gained from
  informing optimization passes about const vs non-const pointers.

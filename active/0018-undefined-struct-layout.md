- Start Date: 2014-05-17
- RFC PR: [rust-lang/rfcs#79](https://github.com/rust-lang/rfcs/pull/79)
- Rust Issue: [rust-lang/rust#14309](https://github.com/rust-lang/rust/issues/14309)

# Summary

Leave structs with unspecified layout by default like enums, for
optimisation purposes. Use something like `#[repr(C)]` to expose C
compatible layout.

# Motivation

The members of a struct are always laid in memory in the order in
which they were specified, e.g.

```rust
struct A {
    x: u8,
    y: u64,
    z: i8,
    w: i64,
}
```

will put the `u8` first in memory, then the `u64`, the `i8` and lastly
the `i64`. Due to the alignment requirements of various types padding
is often required to ensure the members start at an appropriately
aligned byte. Hence the above struct is not `1 + 8 + 1 + 8 == 18`
bytes, but rather `1 + 7 + 8 + 1 + 7 + 8 == 32` bytes, since it is
laid out like

```rust
#[packed] // no automatically inserted padding
struct AFull {
    x: u8,
    _padding1: [u8, .. 7],
    y: u64,
    z: i8,
    _padding2: [u8, .. 7],
    w: i64
}
```

If the fields were reordered to

```rust
struct B {
    y: u64,
    w: i64,

    x: u8,
    i: i8
}
```

then the struct is (strictly) only 18 bytes (but the alignment
requirements of `u64` forces it to take up 24).

Having an undefined layout does allow for possible security
improvements, like randomising struct fields, but this can trivially
be done with a syntax extension that can be attached to a struct to
reorder the fields in the AST itself. That said, there may be benefits
from being able to randomise all structs in a program
automatically/for testing, effectively fuzzing code (especially
`unsafe` code).

Notably, Rust's `enum`s already have undefined layout, and provide the
`#[repr]` attribute to control layout more precisely (specifically,
selecting the size of the discriminant).

# Drawbacks

Forgetting to add `#[repr(C)]` for a struct intended for FFI use can
cause surprising bugs and crashes. There is already a lint for FFI use
of `enum`s without a `#[repr(...)]` attribute, so this can be extended
to include structs.

Having an unspecified (or otherwise non-C-compatible) layout by
default makes interfacing with C slightly harder. A particularly bad
case is passing to C a struct from an upstream library that doesn't
have a `repr(C)` attribute. This situation seems relatively similar to
one where an upstream library type is missing an implementation of a
core trait e.g. `Hash` if one wishes to use it as a hashmap key.

It is slightly better if structs had a specified-but-C-incompatible
layout, *and* one has control over the C interface, because then one
can manually arrange the fields in the C definition to match the Rust
order.

That said, this scenario requires:

- Needing to pass a Rust struct into C/FFI code, where that FFI code
  actually needs to use things from the struct, rather than just pass
  it through, e.g., back into a Rust callback.
- The Rust struct is defined upstream & out of your control, and not
  intended for use with C code.
- The C/FFI code is designed by someone other than that vendor, or
  otherwise not designed for use with the Rust struct (or else it is a
  bug in the vendor's library that the Rust struct can't be sanely
  passed to C).


# Detailed design

A struct declaration like

```rust
struct Foo {
    x: T,
    y: U,
    ...
}
```

has no fixed layout, that is, a compiler can choose whichever order of
fields it prefers.

A fixed layout can be selected with the `#[repr]` attribute

```rust
#[repr(C)]
struct Foo {
    x: T,
    y: U,
    ...
}
```

This will force a struct to be laid out like the equivalent definition
in C.

There would be a lint for the use of non-`repr(C)` structs in related
FFI definitions, for example:

```rust
struct UnspecifiedLayout {
   // ...
}

#[repr(C)]
struct CLayout {
   // ...
}


extern {
    fn foo(x: UnspecifiedLayout); // warning: use of non-FFI-safe struct in extern declaration

    fn bar(x: CLayout); // no warning
}

extern "C" fn foo(x: UnspecifiedLayout) { } // warning: use of non-FFI-safe struct in function with C abi.
```


# Alternatives

- Have non-C layouts opt-in, via `#[repr(smallest)]` and
  `#[repr(random)]` (or similar).
- Have layout defined, but not declaration order (like Java(?)), for
  example, from largest field to smallest, so `u8` fields get placed
  last, and `[u8, .. 1000000]` fields get placed first. The `#[repr]`
  attributes would still allow for selecting declaration-order layout.

# Unresolved questions

- How does this interact with binary compatibility of dynamic libraries?
- How does this interact with DST, where some fields have to be at the
  end of a struct? (Just always lay-out unsized fields last?
  (i.e. after monomorphisation if a field was originally marked
  `Sized?` then it needs to be last).)

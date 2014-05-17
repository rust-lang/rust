- Start Date: 2014-05-17
- RFC PR #:
- Rust Issue #:

# Summary

Leave structs with unspecified layout by default like enums, for
optimisation & security purposes. Use something like `#[repr(C)]` to
expose C compatible layout.

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

There is also some security advantage to being able to randomise
struct layouts, for example,
[the Grsecurity suite](http://grsecurity.net/) of security
enhancements to the Linux kernel provides
[`GRKERNSEC_RANDSTRUCT`](http://en.wikibooks.org/wiki/Grsecurity/Appendix/Grsecurity_and_PaX_Configuration_Options#Randomize_layout_of_sensitive_kernel_structures)
which randomises "sensitive kernel data structures" at compile time.

Notably, Rust's `enum`s already have undefined layout, and provide the
`#[repr]` attribute to control layout more precisely (specifically,
selecting the size of the discriminant).

# Drawbacks

Forgetting to add `#[repr(C)]` for a struct intended for FFI use can
cause surprising bugs and crashes. There is already a lint for FFI use
of `enum`s without a `#[repr(...)]` attribute, so this can be extended
to include structs.

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

# Alternatives

- Have non-C layouts opt-in, via `#[repr(smallest)]` and
  `#[repr(random)]` (or similar).
- Have layout defined, but not declaration order (like Java(?)), for
  example, from largest field to smallest, so `u8` fields get placed
  last, and `[u8, .. 1000000]` fields get placed first. The `#[repr]`
  attributes would still allow for selecting declaration-order layout.

# Unresolved questions

- How does this interact with binary compatibility of dynamic libraries?

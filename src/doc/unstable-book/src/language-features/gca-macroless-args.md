# gca_macroless_args

Enables using `#![feature(gca_min_const_items)]` without the `gca!` macro.

The tracking issue for this feature is: [#159006]

[#159006]: https://github.com/rust-lang/rust/issues/159006

------------------------

Warning: This feature is incomplete; its design and syntax may change.

Related features:
- [gca_min_const_items]. See that doc for what the `gca!` is. This feature enables
support for directly represented const arguments without the macro.
- [gca_macroless_items]. For a version of this feature that works for const arguments
as the right hand side of a const item.

[gca_min_const_items]: gca-min-const-items.md
[gca_macroless_items]: gca-macroless-items.md

## Examples

Here is an example from [gca_min_const_items]:

[gca_min_const_items]: gca-min-const-items.md

```rust
#![allow(incomplete_features)]
#![feature(gca_min_const_items)]

trait Bar {
    #[rustc_always_gca]
    const VAL: usize;
    #[rustc_always_gca]
    const VAL2: usize;
}

struct Baz;

impl Bar for Baz {
    const VAL: usize = core::gca!(2);
    const VAL2: usize = core::gca!(const { Self::VAL * 2 });
}

struct Foo<B: Bar> {
    arr1: [usize; core::gca!(B::VAL)],
    arr2: [usize; core::gca!(B::VAL2)],
}
```

Using `#![feature(gca_macroless_args)]` enables you to write the above without the macro:

```rust
#![allow(incomplete_features)]
#![feature(gca_min_const_items, gca_macroless_args)]

trait Bar {
    #[rustc_always_gca]
    const VAL: usize;
    #[rustc_always_gca]
    const VAL2: usize;
}

struct Baz;

impl Bar for Baz {
    // note these still need a macro, macroless for these is `gca_macroless_items`
    const VAL: usize = core::gca!(2);
    const VAL2: usize = core::gca!(const { Self::VAL * 2 });
}

struct Foo<B: Bar> {
    arr1: [usize; B::VAL],
    arr2: [usize; B::VAL2],
}
```

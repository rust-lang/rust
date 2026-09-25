# gca_macroless_args

Enables implementing const items under `#![feature(gca_min_const_items)]` and `#![feature(gca_const_items)]` without the `gca!` macro.

The tracking issue for this feature is: [#162540]

[#162540]: https://github.com/rust-lang/rust/issues/162540

------------------------

Warning: This feature is incomplete; its design and syntax may change.

Related features:
- [gca_min_const_items]. See that doc for what the `gca!` is. This feature enables
support for directly represented const arguments as the rhs of const items without the macro.
- [gca_macroless_args]. For a version of this feature that works for const arguments in other
positions

[gca_min_const_items]: gca-min-const-items.md
[gca_macroless_args]: gca-macroless-args.md

## Examples

Here is an example from [gca_min_const_items]:

[gca_min_const_items]: gca-min-const-items.md

```rust,ignore (needs new solver)
#![allow(incomplete_features)]
#![feature(
    gca_min_const_items,
    gca_const_items,
    gca_macroless_args,
    generic_const_items,
)]

trait Trait {
    const ASSOC<const N: usize>: usize;
}

impl Trait for () {
    const ASSOC<const N: usize>: usize = core::gca!(N);
}

fn foo<const N: usize>() {
    let a: [(); <() as Trait>::ASSOC::<N>]
        = [(); N];
}
```

Using `#![feature(gca_macroless_items)]` enables you to write the above without the macro:

```rust,ignore (needs new solver)
#![allow(incomplete_features)]
#![feature(
    gca_min_const_items,
    gca_const_items,
    gca_macroless_args,
    gca_macroless_items,
    generic_const_items,
)]

trait Trait {
    const ASSOC<const N: usize>: usize;
}

impl Trait for () {
    const ASSOC<const N: usize>: usize = N;
}

fn foo<const N: usize>() {
    let a: [(); <() as Trait>::ASSOC::<N>]
        = [(); N];
}
```

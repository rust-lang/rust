# macroless_generic_const_args

Enables implementing const items under `#![feature(min_generic_const_args)]` and `#![feature(generic_const_args)]` without the `direct_const_arg!` macro.

The tracking issue for this feature is: [#162540]

[#162540]: https://github.com/rust-lang/rust/issues/162540

------------------------

Warning: This feature is incomplete; its design and syntax may change.

Related features:
- [min_generic_const_args]. See that doc for what the `direct_const_arg!` is. This feature enables
support for directly represented const arguments as the rhs of const items without the macro.
- [macroless_generic_const_args]. For a version of this feature that works for const arguments in other
positions

[min_generic_const_args]: min-generic-const-args.md
[macroless_generic_const_args]: macroless-generic-const-args.md

## Examples

Here is an example from [min_generic_const_args]:

[min_generic_const_args]: min-generic-const-args.md

```rust,ignore (needs new solver)
#![allow(incomplete_features)]
#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    generic_const_items,
)]

trait Trait {
    const ASSOC<const N: usize>: usize;
}

impl Trait for () {
    const ASSOC<const N: usize>: usize = core::direct_const_arg!(N);
}

fn foo<const N: usize>() {
    let a: [(); <() as Trait>::ASSOC::<N>]
        = [(); N];
}
```

Using `#![feature(macroless_const_item_generic_const_args)]` enables you to write the above without the macro:

```rust,ignore (needs new solver)
#![allow(incomplete_features)]
#![feature(
    min_generic_const_args,
    generic_const_args,
    macroless_generic_const_args,
    macroless_const_item_generic_const_args,
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

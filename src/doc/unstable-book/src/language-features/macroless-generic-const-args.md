# macroless_generic_const_args

Enables using `#![feature(min_generic_const_args)]` without the `direct_const_arg!` macro.

The tracking issue for this feature is: [#159006]

[#159006]: https://github.com/rust-lang/rust/issues/159006

------------------------

Warning: This feature is incomplete; its design and syntax may change.

Related features: [min_generic_const_args]. See that doc for what the `direct_const_arg!` is. This feature enables
support for directly represented const arguments without the macro.

[min_generic_const_args]: min-generic-const-args.md

## Examples

Here is an example from [min_generic_const_args]:

[min_generic_const_args]: min-generic-const-args.md

```rust
#![allow(incomplete_features)]
#![feature(min_generic_const_args)]

trait Bar {
    type const VAL: usize;
    type const VAL2: usize;
}

struct Baz;

impl Bar for Baz {
    type const VAL: usize = 2;
    type const VAL2: usize = const { Self::VAL * 2 };
}

struct Foo<B: Bar> {
    arr1: [usize; core::direct_const_arg!(B::VAL)],
    arr2: [usize; core::direct_const_arg!(B::VAL2)],
}
```

Using `#![feature(macroless_generic_const_args)]` enables you to write the above without the macro:

```rust
#![allow(incomplete_features)]
#![feature(min_generic_const_args, macroless_generic_const_args)]

trait Bar {
    type const VAL: usize;
    type const VAL2: usize;
}

struct Baz;

impl Bar for Baz {
    type const VAL: usize = 2;
    type const VAL2: usize = const { Self::VAL * 2 };
}

struct Foo<B: Bar> {
    arr1: [usize; B::VAL],
    arr2: [usize; B::VAL2],
}
```

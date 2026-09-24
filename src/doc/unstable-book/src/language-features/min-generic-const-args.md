# min_generic_const_args

Enables the generic const args MVP (paths to direct const items and constructors for ADTs and primitives).

The tracking issue for this feature is: [#132980]

[#132980]: https://github.com/rust-lang/rust/issues/132980

------------------------

Warning: This feature is incomplete; its design and syntax may change.

This feature acts as a minimal alternative to [generic_const_exprs] that allows a smaller subset of functionality,
and uses a different approach for implementation. It is intentionally more restrictive, which helps with avoiding edge
cases that make the `generic_const_exprs` hard to implement properly. See [Feature background][feature_background]
for more details.

Related features: [macroless_generic_const_args], [generic_const_args], [generic_const_items].

[feature_background]: https://github.com/rust-lang/project-const-generics/blob/main/documents/min_const_generics_plan.md
[generic_const_exprs]: generic-const-exprs.md
[macroless_generic_const_args]: macroless-generic-const-args.md
[generic_const_args]: generic-const-args.md
[generic_const_items]: generic-const-items.md

## `gca!` macro

This feature introduces a new macro: `gca!`.

When an expression is used as a generic argument, it is typically lowered as an "anon const", which is an expression
that is opaque to the type system and cannot contain generics. Using `gca!` instead represents the
expression "directly", i.e. without an anon const, in a way that is visible to the type system.

(Note that plain paths to generic parameters are always represented directly, without `gca!`, as this
already works on stable)

See [macroless_generic_const_args] as a feature to disable the requirement of writing `gca!`.

[macroless_generic_const_args]: macroless-generic-const-args.md

## direct const items

This feature introduces a new item kind: consts with a `gca!` right-hand side.
Constants with a direct right-hand side are allowed to be used in type contexts, e.g.:

```compile_fail
#![allow(incomplete_features)]
#![feature(min_generic_const_args)]

const X: usize = core::gca!(1);
const Y: usize = 1;

struct Foo {
    good_arr: [(); core::gca!(X)], // Allowed
    bad_arr: [(); core::gca!(Y)], // Will not compile
}
```

## Examples

```rust
#![allow(incomplete_features)]
#![feature(min_generic_const_args)]

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

Note that with [generic_const_exprs] the same example would look as follows:

```rust
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

trait Bar {
    const VAL: usize;
    const VAL2: usize;
}

struct Baz;

impl Bar for Baz {
    const VAL: usize = 2;
    const VAL2: usize = const { Self::VAL * 2 };
}

struct Foo<B: Bar>
where
    [(); B::VAL]:,
    [(); B::VAL2]:,
{
    arr1: [usize; B::VAL],
    arr2: [usize; B::VAL2],
}
```

Use of const functions is allowed:

```rust
#![allow(incomplete_features)]
#![feature(min_generic_const_args)]

const VAL: usize = 1;

const fn inc(val: usize) -> usize {
    val + 1
}

const INC: usize = core::gca!(const { inc(VAL) });

const ARR: [usize; INC] = [0; INC];
```

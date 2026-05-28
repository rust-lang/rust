# min_generic_const_args

Enables the generic const args MVP (paths to type const items and constructors for ADTs and primitives).

The tracking issue for this feature is: [#132980]

[#132980]: https://github.com/rust-lang/rust/issues/132980

------------------------

Warning: This feature is incomplete; its design and syntax may change.

This feature acts as a minimal alternative to [generic_const_exprs] that allows a smaller subset of functionality,
and uses a different approach for implementation. It is intentionally more restrictive, which helps with avoiding edge
cases that make the `generic_const_exprs` hard to implement properly. See [Feature background][feature_background]
for more details.

Related features: [generic_const_args], [generic_const_items].

[feature_background]: https://github.com/rust-lang/project-const-generics/blob/main/documents/min_const_generics_plan.md
[generic_const_exprs]: generic-const-exprs.md
[generic_const_args]: generic-const-args.md
[generic_const_items]: generic-const-items.md

## `type const` syntax

This feature introduces new syntax: `type const`.
Constants marked as `type const` are allowed to be used in type contexts, e.g.:

```compile_fail
#![allow(incomplete_features)]
#![feature(min_generic_const_args)]

type const X: usize = 1;
const Y: usize = 1;

struct Foo {
    good_arr: [(); X], // Allowed
    bad_arr: [(); Y], // Will not compile, `Y` must be `type const`.
}
```

## Examples

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
    arr1: [usize; B::VAL],
    arr2: [usize; B::VAL2],
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

type const INC: usize = const { inc(VAL) };

const ARR: [usize; INC] = [0; INC];
```

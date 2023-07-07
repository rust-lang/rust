# Constants in the type system

Constants used in the type system are represented as [`ty::Const`].
The variants of their [`ty::ConstKind`] mostly mirror the variants of [`ty::TyKind`]
with the two *additional* variants being `ConstKind::Value` and `ConstKind::Unevaluated`.

## `WithOptConstParam` and dealing with the query system

To typecheck constants used in the type system, we have to know their expected type.
For const arguments in type dependent paths, e.g. `x.foo::<{ 3 + 4 }>()`, we don't know
the expected type for `{ 3 + 4 }` until we are typechecking the containing function.

As we may however have to evaluate that constant during this typecheck, we would get a cycle error.
For more details, you can look at [this document](https://hackmd.io/@rust-const-generics/Bk5GHW-Iq).

## Unevaluated constants

*This section talks about what's happening with `feature(generic_const_exprs)` enabled.
On stable we do not yet supply any generic parameters to anonymous constants,
avoiding most of the issues mentioned here.*

Unless a constant is either a simple literal, e.g. `[u8; 3]` or `foo::<{ 'c' }>()`,
or a generic parameter, e.g. `[u8; N]`, converting a constant to its [`ty::Const`] representation
returns an unevaluated constant. Even fully concrete constants which do not depend on
generic parameters are not evaluated right away.

Anonymous constants are typechecked separately from their containing item, e.g.
```rust
fn foo<const N: usize>() -> [u8; N + 1] {
    [0; N + 1]
}
```
is treated as
```rust
const ANON_CONST_1<const N: usize> = N + 1;
const ANON_CONST_2<const N: usize> = N + 1;
fn foo<const N: usize>() -> [u8; ANON_CONST_1::<N>] {
    [0; ANON_CONST_2::<N>]
}
```

### Unifying constants

For the compiler, `ANON_CONST_1` and `ANON_CONST_2` are completely different, so
we have to somehow look into unevaluated constants to check whether they should
unify.

For this we use [InferCtxt::try_unify_abstract_consts](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/struct.InferCtxt.html#method.try_unify_abstract_consts).
This builds a custom AST for the two inputs from their THIR. This is then used for
the actual comparison.

### Lazy normalization for constants

We do not eagerly evaluate constant as they can be used in the `where`-clauses of their
parent item, for example:

```rust
#[feature(generic_const_exprs)]
fn foo<T: Trait>()
where
    [u8; <T as  Trait>::ASSOC + 1]: SomeOtherTrait,
{}
```

The constant `<T as  Trait>::ASSOC + 1` depends on the `T: Trait` bound of
its parents caller bounds, but is also part of another bound itself.
If we were to eagerly evaluate this constant while computing its parents bounds
this would cause a query cycle.

### Unused generic arguments of anonymous constants

Anonymous constants inherit the generic parameters of their parent, which is
why the array length in `foo<const N: usize>() -> [u8; N + 1]` can use `N`.

Without any manual adjustments, this causes us to include parameters even if
the constant doesn't use them in any way. This can cause
[some interesting errors][pcg-unused-substs] and breaks some already stable code.

[`ty::Const`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Const.html
[`ty::ConstKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/sty/enum.ConstKind.html
[`ty::TyKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/sty/enum.TyKind.html
[pcg-unused-substs]: https://github.com/rust-lang/project-const-generics/issues/33

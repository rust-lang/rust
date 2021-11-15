# Constants in the type system

Constants used in the type system are represented as [`ty::Const`].
The variants of their [`ty::ConstKind`] mostly mirror the variants of [`ty::TyKind`]
with the two *additional* variants being `ConstKind::Value` and `ConstKind::Unevaluated`.


## Unevaluated constants

*This section talks about what's happening with `feature(generic_const_exprs)` enabled.
On stable we do not yet supply any generic parameters to anonymous constants,
avoiding most of the issues mentioned here.*

Unless a constant is either a simple literal, e.g. `[u8; 3]` or `foo::<{ 'c' }>()`,
or a generic parameter, e.g. `[u8; N]`, converting a constant to its [`ty::Const`] representation
returns an unevaluated constant. Even fully concrete constants which do not depend on
generic parameters are not evaluated right away.

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

### Generic arguments of anonymous constants

Anonymous constants inherit the generic parameters of their parent, which is
why the array length in `foo<const N: usize>() -> [u8; N + 1]` can use `N`.

Without any manual adjustments, this causes us to include parameters even if
the constant doesn't use them in any way. This can cause
[some interesting errors][pcg-unused-substs] and breaks some already stable code.

To deal with this, we intend to look at the generic parameters explicitly mentioned
by the constants and then search the predicates of its parents to figure out which
of the other generic parameters are reachable by our constant.

**TODO**: Expand this section once the parameter filtering is implemented.

As constants can be part of their parents `where`-clauses, we mention unevaluated
constants in their parents predicates. It is therefore necessary to mention unevaluated
constants before we have computed the generic parameters
available to these constants.

To do this unevaluated constants start out with [`substs_`] being `None` while assuming
that their generic arguments could be arbitrary generic parameters.
When first accessing the generic arguments of an unevaluated constants, we then replace
`substs_` with the actual default arguments of a constants, which are the generic parameters
of their parent we assume to be used by this constant.

[`ty::Const`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Const.html
[`ty::ConstKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.ConstKind.html
[`ty::TyKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/enum.TyKind.html
[pcg-unused-substs]: https://github.com/rust-lang/project-const-generics/blob/master/design-docs/anon-const-substs.md#unused-substs
[`substs_`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/consts/kind/struct.Unevaluated.html#structfield.substs_
# Ambig/Unambig Types and Consts

Types and Consts args in the HIR can be in two kinds of positions ambiguous (ambig) or unambiguous (unambig). Ambig positions are where
it would be valid to parse either a type or a const, unambig positions are where only one kind would be valid to
parse.

```rust
fn func<T, const N: usize>(arg: T) {
    //                           ^ Unambig type position
    let a: _ = arg; 
    //     ^ Unambig type position

    func::<T, N>(arg);
    //     ^  ^
    //     ^^^^ Ambig position 

    let _: [u8; 10];
    //      ^^  ^^ Unambig const position
    //      ^^ Unambig type position
}

```

Most types/consts in ambig positions are able to be disambiguated as either a type or const during parsing. Single segment paths are always represented as types in the AST but may get resolved to a const parameter during name resolution, then lowered to a const argument during ast-lowering. The only generic arguments which remain ambiguous after lowering are inferred generic arguments (`_`) in path segments. For example, in `Foo<_>` it is not clear whether the `_` argument is an inferred type argument, or an inferred const argument.

In unambig positions, inferred arguments are represented with [`hir::TyKind::Infer`][ty_infer] or [`hir::ConstArgKind::Infer`][const_infer] depending on whether it is a type or const position respectively.
In ambig positions, inferred arguments are represented with `hir::GenericArg::Infer`.

A naive implementation of this would result in there being potentially 5 places where you might think an inferred type/const could be found in the HIR from looking at the structure of the HIR:
1. In unambig type position as a `hir::TyKind::Infer`
2. In unambig const arg position as a `hir::ConstArgKind::Infer`
3. In an ambig position as a [`GenericArg::Type(TyKind::Infer)`][generic_arg_ty]
4. In an ambig position as a [`GenericArg::Const(ConstArgKind::Infer)`][generic_arg_const]
5. In an ambig position as a [`GenericArg::Infer`][generic_arg_infer]

Note that places 3 and 4 would never actually be possible to encounter as we always lower to `GenericArg::Infer` in generic arg position. 

This has a few failure modes:
- People may write visitors which check for `GenericArg::Infer` but forget to check for `hir::TyKind/ConstArgKind::Infer`, only handling infers in ambig positions by accident.
- People may write visitors which check for `hir::TyKind/ConstArgKind::Infer` but forget to check for `GenericArg::Infer`, only handling infers in unambig positions by accident.
- People may write visitors which check for `GenericArg::Type/Const(TyKind/ConstArgKind::Infer)` and `GenericArg::Infer`, not realising that we never represent inferred types/consts in ambig positions as a `GenericArg::Type/Const`.
- People may write visitors which check for *only* `TyKind::Infer` and not `ConstArgKind::Infer` forgetting that there are also inferred const arguments (and vice versa).

To make writing HIR visitors less error prone when caring about inferred types/consts we have a relatively complex system:

1. We have different types in the compiler for when a type or const is in an unambig or ambig position, `hir::Ty<AmbigArg>` and `hir::Ty<()>`. [`AmbigArg`][ambig_arg] is an uninhabited type which we use in the `Infer` variant of `TyKind` and `ConstArgKind` to selectively "disable" it if we are in an ambig position.

2. The [`visit_ty`][visit_ty] and [`visit_const_arg`][visit_const_arg] methods on HIR visitors only accept the ambig position versions of types/consts. Unambig types/consts are implicitly converted to ambig types/consts during the visiting process, with the `Infer` variant handled by a dedicated [`visit_infer`][visit_infer] method.

This has a number of benefits:
- It's clear that `GenericArg::Type/Const` cannot represent inferred type/const arguments
- Implementors of `visit_ty` and `visit_const_arg` will never encounter inferred types/consts making it impossible to write a visitor that seems to work right but handles edge cases wrong 
- The `visit_infer` method handles *all* cases of inferred type/consts in the HIR making it easy for visitors to handle inferred type/consts in one dedicated place and not forget cases

[ty_infer]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.TyKind.html#variant.Infer
[const_infer]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.ConstArgKind.html#variant.Infer
[generic_arg_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.GenericArg.html#variant.Type
[generic_arg_const]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.GenericArg.html#variant.Const
[generic_arg_infer]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.GenericArg.html#variant.Infer
[ambig_arg]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.AmbigArg.html
[visit_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/intravisit/trait.Visitor.html#method.visit_ty
[visit_const_arg]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/intravisit/trait.Visitor.html#method.visit_const_arg
[visit_infer]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/intravisit/trait.Visitor.html#method.visit_infer

# Ambig/Unambig Types and Consts

Types and Consts args in the AST/HIR can be in two kinds of positions ambiguous (ambig) or unambiguous (unambig). Ambig positions are where
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

Most types/consts in ambig positions are able to be disambiguated as either a type or const during parsing. The only exceptions to this are paths and inferred generic arguments.

## Paths

```rust
struct Foo<const N: usize>;

fn foo<const N: usize>(_: Foo<N>) {}
```

At parse time we parse all unbraced generic arguments as *types* (ie they wind up as [`ast::GenericArg::Ty`]). In the above example this means we would parse the generic argument to `Foo` as an `ast::GenericArg::Ty` wrapping a [`ast::Ty::Path(N)`].

Then during name resolution:
- When encountering a single segment path with no generic arguments in generic argument position, we will first try to resolve it in the type namespace and if that fails we then attempt to resolve in the value namespace.
- All other kinds of paths we only try to resolve in the type namespace

See [`LateResolutionVisitor::visit_generic_arg`] for where this is implemented.

Finally during AST lowering when we attempt to lower a type argument, we first check if it is a `Ty::Path` and if it resolved to something in the value namespace. If it did then we create an *anon const* and lower to a const argument instead of a type argument.

See [`LoweringContext::lower_generic_arg`] for where this is implemented.

Note that the ambiguity for paths is not propgated into the HIR; there's no `hir::GenericArg::Path` which is turned into either a `Ty` or `Const` during HIR ty lowering (though we could do such a thing).

## Inferred arguments (`_`)

```rust
struct Foo<const N: usize>;

fn foo() {
    let _unused: Foo<_>;
}
```

The only generic arguments which remain ambiguous after lowering are inferred generic arguments (`_`) in path segments. In the above example it is not clear at parse time whether the `_` argument to `Foo` is an inferred type argument, or an inferred const argument.

In ambig AST positions, inferred argumentsd are parsed as an [`ast::GenericArg::Ty`] wrapping a [`ast::Ty::Infer`]. Then during AST lowering when lowering an `ast::GenericArg::Ty` we check if it is an inferred type and if so lower to a [`hir::GenericArg::Infer`].

In unambig AST positions, inferred arguments are parsed as either `ast::Ty::Infer` or [`ast::AnonConst`]. The `AnonConst` case is quite strange, we use [`ast::ExprKind::Underscore`] to represent the "body" of the "anon const" although in reality we do not actually lower this to an anon const in the HIR.

It may be worth seeing if we can refactor the AST to have `ast::GenericArg::Infer` and then get rid of this overloaded meaning of `AnonConst`, as well as the reuse of `ast::Ty::Infer` in ambig positions.

In unambig AST positions, during AST lowering we lower inferred arguments to [`hir::TyKind::Infer`][ty_infer] or [`hir::ConstArgKind::Infer`][const_infer] depending on whether it is a type or const position respectively.
In ambig AST positions, during AST lowering we lower inferred arguments to [`hir::GenericArg::Infer`][generic_arg_infer]. See [`LoweringContext::lower_generic_arg`] for where this is implemented.

A naive implementation of this would result in there being potentially 5 places where you might think an inferred type/const could be found in the HIR from looking at the structure of the HIR:
1. In unambig type position as a `TyKind::Infer`
2. In unambig const arg position as a `ConstArgKind::Infer`
3. In an ambig position as a [`GenericArg::Type(TyKind::Infer)`][generic_arg_ty]
4. In an ambig position as a [`GenericArg::Const(ConstArgKind::Infer)`][generic_arg_const]
5. In an ambig position as a `GenericArg::Infer`

Note that places 3 and 4 would never actually be possible to encounter as we always lower to `GenericArg::Infer` in generic arg position. 

This has a few failure modes:
- People may write visitors which check for `GenericArg::Infer` but forget to check for `hir::TyKind/ConstArgKind::Infer`, only handling infers in ambig positions by accident.
- People may write visitors which check for `TyKind/ConstArgKind::Infer` but forget to check for `GenericArg::Infer`, only handling infers in unambig positions by accident.
- People may write visitors which check for `GenericArg::Type/Const(TyKind/ConstArgKind::Infer)` and `GenericArg::Infer`, not realising that we never represent inferred types/consts in ambig positions as a `GenericArg::Type/Const`.
- People may write visitors which check for *only* `TyKind::Infer` and not `ConstArgKind::Infer` forgetting that there are also inferred const arguments (and vice versa).

To make writing HIR visitors less error prone when caring about inferred types/consts we have a relatively complex system:

1. We have different types in the compiler for when a type or const is in an unambig or ambig position, `Ty<AmbigArg>` and `Ty<()>`. [`AmbigArg`][ambig_arg] is an uninhabited type which we use in the `Infer` variant of `TyKind` and `ConstArgKind` to selectively "disable" it if we are in an ambig position.

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
[`LateResolutionVisitor::visit_generic_arg`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_resolve/late/struct.LateResolutionVisitor.html#method.visit_generic_arg
[`LoweringContext::lower_generic_arg`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast_lowering/struct.LoweringContext.html#method.lower_generic_arg
[`ast::GenericArg::Ty`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/enum.GenericArg.html#variant.Type
[`ast::Ty::Infer`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/enum.TyKind.html#variant.Infer
[`ast::AnonConst`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/struct.AnonConst.html
[`hir::GenericArg::Infer`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.GenericArg.html#variant.Infer
[`ast::ExprKind::Underscore`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/enum.ExprKind.html#variant.Underscore
[`ast::Ty::Path(N)`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/enum.TyKind.html#variant.Path
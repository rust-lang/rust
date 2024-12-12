use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::symbol::{kw, sym};

use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `default_could_be_derived` lint checks for manual `impl` blocks
    /// of the `Default` trait that could have been derived.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// struct A {
    ///     b: Option<i32>,
    /// }
    ///
    /// #[deny(default_could_be_derived)]
    /// impl Default for Foo {
    ///     fn default() -> Foo {
    ///         A {
    ///             b: None,
    ///         }
    ///     }
    /// }
    ///
    /// enum Foo {
    ///     Bar,
    /// }
    ///
    /// #[deny(default_could_be_derived)]
    /// impl Default for Foo {
    ///     fn default() -> Foo {
    ///         Foo::Bar
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `#[derive(Default)]` uses the `Default` impl for every field of your
    /// type. If your manual `Default` impl either invokes `Default::default()`
    /// or uses the same value that that associated function produces, then it
    /// is better to use the derive to avoid the different `Default` impls from
    /// diverging over time.
    ///
    /// This lint also triggers on cases where there the type has no fields,
    /// so the derive for `Default` for a struct is trivial, and for an enum
    /// variant with no fields, which can be annotated with `#[default]`.
    pub DEFAULT_COULD_BE_DERIVED,
    Deny,
    "detect `Default` impl that could be derived"
}

declare_lint! {
    /// The `default_could_be_derived` lint checks for manual `impl` blocks
    /// of the `Default` trait that could have been derived.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// struct Foo {
    ///     x: i32 = 101,
    /// }
    ///
    /// #[deny(manual_default_for_type_with_default_fields)]
    /// impl Default for Foo {
    ///     fn default() -> Foo {
    ///         Foo { x: 100 }
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Manually writing a `Default` implementation for a type that has
    /// default field values runs the risk of diverging behavior between
    /// `Type { .. }` and `<Type as Default>::default()`, which would be a
    /// foot-gun for users of that type that would expect these to be
    /// equivalent.
    pub MANUAL_DEFAULT_FOR_TYPE_WITH_DEFAULT_FIELDS,
    Warn,
    "detect `Default` impl on type with default field values that should be derived"
}

declare_lint_pass!(DefaultCouldBeDerived => [DEFAULT_COULD_BE_DERIVED, MANUAL_DEFAULT_FOR_TYPE_WITH_DEFAULT_FIELDS]);

impl<'tcx> LateLintPass<'tcx> for DefaultCouldBeDerived {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let hir = cx.tcx.hir();
        let hir::ItemKind::Impl(data) = item.kind else { return };
        let Some(trait_ref) = data.of_trait else { return };
        let Res::Def(DefKind::Trait, def_id) = trait_ref.path.res else { return };
        let Some(default_def_id) = cx.tcx.get_diagnostic_item(sym::Default) else { return };
        if Some(def_id) != Some(default_def_id) {
            return;
        }
        if cx.tcx.has_attr(def_id, sym::automatically_derived) {
            return;
        }
        let hir_self_ty = data.self_ty;
        let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = hir_self_ty.kind else { return };
        let Res::Def(def_kind, type_def_id) = path.res else { return };
        match hir.get_if_local(type_def_id) {
            Some(hir::Node::Item(hir::Item {
                kind:
                    hir::ItemKind::Struct(hir::VariantData::Struct { fields, recovered: _ }, _generics),
                ..
            })) => {
                let fields_with_default_value: Vec<_> =
                    fields.iter().filter_map(|f| f.default).collect();
                let fields_with_default_impl: Vec<_> = fields
                    .iter()
                    .filter_map(|f| match (f.ty.kind, f.default) {
                        (hir::TyKind::Path(hir::QPath::Resolved(_, path)), None)
                            if let Some(def_id) = path.res.opt_def_id()
                                && let DefKind::Struct | DefKind::Enum =
                                    cx.tcx.def_kind(def_id) =>
                        {
                            let ty = cx.tcx.type_of(def_id).instantiate_identity();
                            let mut count = 0;
                            cx.tcx.for_each_relevant_impl(default_def_id, ty, |_| count += 1);
                            if count > 0 { Some(f.ty.span) } else { None }
                        }
                        _ => None,
                    })
                    .collect();
                if !fields_with_default_value.is_empty()
                    && fields.len()
                        == fields_with_default_value.len() + fields_with_default_impl.len()
                {
                    cx.tcx.node_span_lint(
                        MANUAL_DEFAULT_FOR_TYPE_WITH_DEFAULT_FIELDS,
                        item.hir_id(),
                        item.span,
                        |diag| {
                            diag.primary_message(
                                "manual `Default` impl for type with default field values",
                            );
                            let msg = match (
                                !fields_with_default_value.is_empty(),
                                !fields_with_default_impl.is_empty(),
                            ) {
                                (true, true) => "default values or a type that impls `Default`",
                                (true, false) => "default values",
                                (false, true) => "a type that impls `Default`",
                                (false, false) => unreachable!(),
                            };
                            diag.span_label(
                                cx.tcx.def_span(type_def_id),
                                format!("all the fields in this struct have {msg}"),
                            );
                            for anon in fields_with_default_value {
                                diag.span_label(anon.span, "default value");
                            }
                            for field in fields_with_default_impl {
                                diag.span_label(field, "implements `Default`");
                            }
                            diag.multipart_suggestion_verbose(
                                "to avoid divergence in behavior between `Struct { .. }` and \
                                 `<Struct as Default>::default()`, derive the `Default`",
                                vec![
                                    (
                                        cx.tcx.def_span(type_def_id).shrink_to_lo(),
                                        "#[derive(Default)] ".to_string(),
                                    ),
                                    (item.span, String::new()),
                                ],
                                Applicability::MachineApplicable,
                            );
                        },
                    );
                    return;
                }
            }
            _ => {}
        }
        let generics = cx.tcx.generics_of(type_def_id);
        if !generics.own_params.is_empty() && def_kind != DefKind::Enum {
            // For enums, `#[derive(Default)]` forces you to select a unit variant to avoid
            // "imperfect derives", unnecessary bounds on type parameters, so even if the enum has
            // type parameters we can still lint on the manual impl if the return is a unit
            // variant.
            return;
        }
        // We have a manual `impl Default for Ty {}` item, where `Ty` has no type parameters.

        for assoc in data.items {
            // We look for the user's `fn default() -> Self` associated fn of the `Default` impl.

            let hir::AssocItemKind::Fn { has_self: false } = assoc.kind else { continue };
            if assoc.ident.name != kw::Default {
                continue;
            }
            let assoc = hir.impl_item(assoc.id);
            let hir::ImplItemKind::Fn(_ty, body) = assoc.kind else { continue };
            let body = hir.body(body);
            let hir::ExprKind::Block(hir::Block { stmts: [], expr: Some(expr), .. }, None) =
                body.value.kind
            else {
                continue;
            };

            // We check `fn default()` body is a single ADT literal and all the fields are being
            // set to something equivalent to the corresponding types' `Default::default()`.
            match expr.kind {
                hir::ExprKind::Path(hir::QPath::Resolved(_, path))
                    if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), def_id) =
                        path.res =>
                {
                    // We have a unit variant as the default of an enum in a manual impl.
                    //
                    // enum Foo {
                    //     Bar,
                    // }
                    //
                    // impl Default for Foo {
                    //     fn default() -> Foo {
                    //         Foo::Bar
                    //     }
                    // }
                    //
                    // We suggest
                    //
                    // #[derive(Default)] enum Foo {
                    //     #[default] Bar,
                    // }
                    cx.tcx.node_span_lint(
                        DEFAULT_COULD_BE_DERIVED,
                        item.hir_id(),
                        item.span,
                        |diag| {
                            diag.primary_message("`impl Default` that could be derived");
                            diag.span_label(
                                expr.span,
                                "this enum variant has no fields, so it's trivially derivable",
                            );
                            diag.multipart_suggestion_verbose(
                                "you don't need to manually `impl Default`, you can derive it",
                                vec![
                                    (
                                        cx.tcx.def_span(type_def_id).shrink_to_lo(),
                                        "#[derive(Default)] ".to_string(),
                                    ),
                                    (
                                        cx.tcx.def_span(def_id).shrink_to_lo(),
                                        "#[default] ".to_string(),
                                    ),
                                    (item.span, String::new()),
                                ],
                                Applicability::MachineApplicable,
                            );
                        },
                    );
                }
                hir::ExprKind::Struct(_qpath, fields, tail) => {
                    // We have a struct literal
                    //
                    // struct Foo {
                    //     field: Type,
                    // }
                    //
                    // impl Default for Foo {
                    //     fn default() -> Foo {
                    //         Foo {
                    //             field: val,
                    //         }
                    //     }
                    // }
                    //
                    // We suggest #[derive(Default)] if
                    //  - `val` is `Default::default()`
                    //  - `val` matches the `Default::default()` body for that type
                    //  - `val` is `0`
                    //  - `val` is `false`
                    if let hir::StructTailExpr::Base(_) = tail {
                        // This is *very* niche. We'd only get here if someone wrote
                        // impl Default for Ty {
                        //     fn default() -> Ty {
                        //         Ty { ..something() }
                        //     }
                        // }
                        // where `something()` would have to be a call or path.
                        return;
                    }
                    if fields.iter().all(|f| check_expr(cx, f.expr)) {
                        cx.tcx.node_span_lint(
                            DEFAULT_COULD_BE_DERIVED,
                            item.hir_id(),
                            item.span,
                            |diag| {
                                diag.primary_message("`impl Default` that could be derived");
                                for (i, field) in fields.iter().enumerate() {
                                    let msg = if i == fields.len() - 1 {
                                        if fields.len() == 1 {
                                            "this is the same value the expansion of \
                                             `#[derive(Default)]` would use"
                                        } else {
                                            "these are the same values the expansion of \
                                             `#[derive(Default)]` would use"
                                        }
                                    } else {
                                        ""
                                    };
                                    diag.span_label(field.expr.span, msg);
                                }
                                if let hir::StructTailExpr::DefaultFields(span) = tail {
                                    diag.span_label(
                                        span,
                                        "all remaining fields will use the same default field \
                                         values that the `#[derive(Default)]` would use",
                                    );
                                }
                                diag.multipart_suggestion_verbose(
                                    "you don't need to manually `impl Default`, you can derive it",
                                    vec![
                                        (
                                            cx.tcx.def_span(type_def_id).shrink_to_lo(),
                                            "#[derive(Default)] ".to_string(),
                                        ),
                                        (item.span, String::new()),
                                    ],
                                    Applicability::MachineApplicable,
                                );
                            },
                        );
                    }
                }
                hir::ExprKind::Call(expr, args) => {
                    if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = expr.kind
                        && let Res::Def(DefKind::Ctor(CtorOf::Struct, CtorKind::Fn), ctor_def_id) =
                            path.res
                    {
                        let type_def_id = cx.tcx.parent(ctor_def_id); // From Ctor to struct
                        if args.iter().all(|expr| check_expr(cx, expr)) {
                            // We have a struct literal
                            //
                            // struct Foo(Type);
                            //
                            // impl Default for Foo {
                            //     fn default() -> Foo {
                            //         Foo(val)
                            //     }
                            // }
                            //
                            // We suggest #[derive(Default)] if
                            //  - `val` is `Default::default()`
                            //  - `val` matches the `Default::default()` body for that type
                            //  - `val` is `0`
                            //  - `val` is `false`
                            cx.tcx.node_span_lint(
                                DEFAULT_COULD_BE_DERIVED,
                                item.hir_id(),
                                item.span,
                                |diag| {
                                    diag.primary_message("`impl Default` that could be derived");
                                    for (i, field) in args.iter().enumerate() {
                                        let msg = if i == args.len() - 1 {
                                            if args.len() == 1 {
                                                "this is the same value the expansion of \
                                                 `#[derive(Default)]` would use"
                                            } else {
                                                "these are the same values the expansion of \
                                                 `#[derive(Default)]` would use"
                                            }
                                        } else {
                                            ""
                                        };
                                        diag.span_label(field.span, msg);
                                    }
                                    diag.multipart_suggestion_verbose(
                                        "you don't need to manually `impl Default`, you can derive it",
                                        vec![
                                            (
                                                cx.tcx.def_span(type_def_id).shrink_to_lo(),
                                                "#[derive(Default)] ".to_string(),
                                            ),
                                            (item.span, String::new()),
                                        ],
                                        Applicability::MachineApplicable,
                                    );
                                },
                            );
                        }
                    }
                }
                hir::ExprKind::Path(hir::QPath::Resolved(_, path))
                    if let Res::Def(DefKind::Ctor(CtorOf::Struct, CtorKind::Const), _) =
                        path.res =>
                {
                    // We have a struct literal
                    //
                    // struct Foo;
                    //
                    // impl Default for Foo {
                    //     fn default() -> Foo {
                    //         Foo
                    //     }
                    // }
                    //
                    // We always suggest `#[derive(Default)]`.
                    cx.tcx.node_span_lint(
                        DEFAULT_COULD_BE_DERIVED,
                        item.hir_id(),
                        item.span,
                        |diag| {
                            diag.primary_message("`impl Default` that could be derived");
                            diag.span_label(
                                expr.span,
                                "this type has no fields, so it's trivially derivable",
                            );
                            diag.multipart_suggestion_verbose(
                                "you don't need to manually `impl Default`, you can derive it",
                                vec![
                                    (
                                        cx.tcx.def_span(type_def_id).shrink_to_lo(),
                                        "#[derive(Default)] ".to_string(),
                                    ),
                                    (item.span, String::new()),
                                ],
                                Applicability::MachineApplicable,
                            );
                        },
                    );
                }
                _ => {}
            }
        }
    }
}

fn check_path<'tcx>(
    cx: &LateContext<'tcx>,
    path: &hir::QPath<'_>,
    hir_id: hir::HirId,
    ty: Ty<'tcx>,
) -> bool {
    let Some(default_def_id) = cx.tcx.get_diagnostic_item(sym::Default) else {
        return false;
    };
    let res = cx.qpath_res(&path, hir_id);
    let Some(def_id) = res.opt_def_id() else { return false };
    if cx.tcx.is_diagnostic_item(sym::default_fn, def_id) {
        // We have `field: Default::default(),`. This is what the derive would do already.
        return true;
    }
    // For every `Default` impl for this type (there should be a single one), we see if it
    // has a "canonical" `DefId` for a fn call with no arguments, or a path. If it does, we
    // check that `DefId` with the `DefId` of this field's value if it is also a call/path.
    // If there's a match, it means that the contents of that type's `Default` impl are the
    // same to what the user wrote on *their* `Default` impl for this field.
    let mut equivalents = vec![];
    cx.tcx.for_each_relevant_impl(default_def_id, ty, |impl_def_id| {
        let equivalent = match impl_def_id.as_local() {
            None => cx.tcx.get_default_impl_equivalent(impl_def_id),
            Some(local) => {
                let def_kind = cx.tcx.def_kind(impl_def_id);
                cx.tcx.get_default_equivalent(def_kind, local)
            }
        };
        if let Some(did) = equivalent {
            equivalents.push(did);
        }
    });
    for did in equivalents {
        if did == def_id {
            return true;
        }
    }
    false
}

fn check_expr(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    match expr.kind {
        hir::ExprKind::Lit(spanned_lit) => match spanned_lit.node {
            LitKind::Int(val, _) if val == 0 => true, // field: 0,
            LitKind::Bool(false) => true,             // field: false,
            _ => false,
        },
        hir::ExprKind::Call(hir::Expr { kind: hir::ExprKind::Path(path), hir_id, .. }, []) => {
            // `field: foo(),` or `field: Ty::assoc(),`
            let Some(ty) = cx
                .tcx
                .has_typeck_results(expr.hir_id.owner.def_id)
                .then(|| cx.tcx.typeck(expr.hir_id.owner.def_id))
                .and_then(|typeck| typeck.expr_ty_adjusted_opt(expr))
            else {
                return false;
            };
            check_path(cx, &path, *hir_id, ty)
        }
        hir::ExprKind::Path(path) => {
            // `field: qualified::Path,` or `field: <Ty as Trait>::Assoc,`
            let Some(ty) = cx
                .tcx
                .has_typeck_results(expr.hir_id.owner.def_id)
                .then(|| cx.tcx.typeck(expr.hir_id.owner.def_id))
                .and_then(|typeck| typeck.expr_ty_adjusted_opt(expr))
            else {
                return false;
            };
            check_path(cx, &path, expr.hir_id, ty)
        }
        _ => false,
    }
}

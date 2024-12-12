use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_middle::ty::TyCtxt;
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
    pub DEFAULT_COULD_BE_DERIVED,
    Warn,
    "detect `Default` impl that could be derived"
}

declare_lint_pass!(DefaultCouldBeDerived => [DEFAULT_COULD_BE_DERIVED]);

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
                        DEFAULT_COULD_BE_DERIVED,
                        item.hir_id(),
                        item.span,
                        |diag| {
                            diag.primary_message("`impl Default` that could be derived");
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
                hir::ExprKind::Struct(_qpath, fields, _tail) => {
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
                    //  - `val` is `0`
                    //  - `val` is `false`
                    if fields.iter().all(|f| check_expr(cx.tcx, f.expr.kind)) {
                        cx.tcx.node_span_lint(
                            DEFAULT_COULD_BE_DERIVED,
                            item.hir_id(),
                            item.span,
                            |diag| {
                                diag.primary_message("`impl Default` that could be derived");
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
                        if args.iter().all(|expr| check_expr(cx.tcx, expr.kind)) {
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
                            //  - `val` is `0`
                            //  - `val` is `false`
                            cx.tcx.node_span_lint(
                                DEFAULT_COULD_BE_DERIVED,
                                item.hir_id(),
                                item.span,
                                |diag| {
                                    diag.primary_message("`impl Default` that could be derived");
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

fn check_expr(tcx: TyCtxt<'_>, kind: hir::ExprKind<'_>) -> bool {
    let Some(default_def_id) = tcx.get_diagnostic_item(sym::Default) else {
        return false;
    };
    match kind {
        hir::ExprKind::Lit(spanned_lit) => match spanned_lit.node {
            LitKind::Int(val, _) if val == 0 => true, // field: 0,
            LitKind::Bool(false) => true,             // field: false,
            _ => false,
        },
        hir::ExprKind::Call(expr, [])
            if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = expr.kind
                && let Some(def_id) = path.res.opt_def_id()
                && tcx.is_diagnostic_item(sym::default_fn, def_id) =>
        {
            // field: Default::default(),
            true
        }
        hir::ExprKind::Path(hir::QPath::Resolved(_, path))
            if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), ctor_def_id) =
                path.res =>
        {
            // FIXME: We should use a better check where we explore existing
            // `impl Default for def_id` of the found type when `def_id` is not
            // local and see compare them against what we have here. For now,
            // we special case `Option::None` and only check unit variants of
            // local `Default` impls.
            let var_def_id = tcx.parent(ctor_def_id); // From Ctor to variant

            // We explicitly check for `Option::<T>::None`. If `Option` was
            // local, it would be accounted by the logic further down, but
            // because the analysis uses purely the HIR, that doesn't work
            // accross crates.
            //
            // field: None,
            let mut found = tcx.is_lang_item(var_def_id, hir::LangItem::OptionNone);

            // Look at the local `impl Default for ty` of the field's `ty`.
            let ty_def_id = tcx.parent(var_def_id); // From variant to enum
            let ty = tcx.type_of(ty_def_id).instantiate_identity();
            tcx.for_each_relevant_impl(default_def_id, ty, |impl_did| {
                let hir = tcx.hir();
                let Some(hir::Node::Item(impl_item)) = hir.get_if_local(impl_did) else {
                    return;
                };
                let hir::ItemKind::Impl(impl_item) = impl_item.kind else {
                    return;
                };
                for assoc in impl_item.items {
                    let hir::AssocItemKind::Fn { has_self: false } = assoc.kind else {
                        continue;
                    };
                    if assoc.ident.name != kw::Default {
                        continue;
                    }
                    let assoc = hir.impl_item(assoc.id);
                    let hir::ImplItemKind::Fn(_ty, body) = assoc.kind else {
                        continue;
                    };
                    let body = hir.body(body);
                    let hir::ExprKind::Block(hir::Block { stmts: [], expr: Some(expr), .. }, None) =
                        body.value.kind
                    else {
                        continue;
                    };
                    // Look at a specific implementation of `Default::default()`
                    // for their content and see if they are requivalent to what
                    // the user wrote in their manual `impl` for a given field.
                    match expr.kind {
                        hir::ExprKind::Path(hir::QPath::Resolved(_, path))
                            if let Res::Def(
                                DefKind::Ctor(CtorOf::Variant, CtorKind::Const),
                                orig_def_id,
                            ) = path.res =>
                        {
                            // We found
                            //
                            // field: Foo::Unit,
                            //
                            // and
                            //
                            // impl Default for Foo {
                            //     fn default() -> Foo { Foo::Unit }
                            // }
                            found |= orig_def_id == ctor_def_id
                        }
                        _ => {}
                    }
                }
            });
            found
        }
        _ => false,
    }
}

use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
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
        let hir::ItemKind::Impl(data) = item.kind else { return };
        let Some(trait_ref) = data.of_trait else { return };
        let Res::Def(DefKind::Trait, def_id) = trait_ref.path.res else { return };
        if Some(def_id) != cx.tcx.get_diagnostic_item(sym::Default) {
            return;
        }
        if cx.tcx.has_attr(def_id, sym::automatically_derived) {
            return;
        }
        let hir_self_ty = data.self_ty;
        let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = hir_self_ty.kind else { return };
        let Res::Def(_, type_def_id) = path.res else { return };
        let generics = cx.tcx.generics_of(type_def_id);
        if !generics.own_params.is_empty() {
            return;
        }
        // We have a manual `impl Default for Ty {}` item, where `Ty` has no type parameters.

        let hir = cx.tcx.hir();
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
                _ => {}
            }
        }
    }
}

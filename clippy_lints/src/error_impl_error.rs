use clippy_utils::{
    diagnostics::{span_lint, span_lint_hir_and_then},
    path_res,
    ty::implements_trait,
};
use rustc_hir::{def_id::DefId, Item, ItemKind, Node};
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for types named `Error` that implement `Error`.
    ///
    /// ### Why is this bad?
    /// It can become confusing when a codebase has 20 types all named `Error`, requiring either
    /// aliasing them in the `use` statement them or qualifying them like `my_module::Error`. This
    /// severely hinders readability.
    ///
    /// ### Example
    /// ```rust,ignore
    /// #[derive(Debug)]
    /// pub enum Error { ... }
    ///
    /// impl std::fmt::Display for Error { ... }
    ///
    /// impl std::error::Error for Error { ... }
    /// ```
    #[clippy::version = "1.72.0"]
    pub ERROR_IMPL_ERROR,
    restriction,
    "types named `Error` that implement `Error`"
}
declare_lint_pass!(ErrorImplError => [ERROR_IMPL_ERROR]);

impl<'tcx> LateLintPass<'tcx> for ErrorImplError {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        let Some(error_def_id) = cx.tcx.get_diagnostic_item(sym::Error) else {
            return;
        };

        match item.kind {
            ItemKind::TyAlias(ty, _) if implements_trait(cx, hir_ty_to_ty(cx.tcx, ty), error_def_id, &[])
                && item.ident.name == sym::Error =>
            {
                span_lint(
                    cx,
                    ERROR_IMPL_ERROR,
                    item.ident.span,
                    "type alias named `Error` that implements `Error`",
                );
            },
            ItemKind::Impl(imp) if let Some(trait_def_id) = imp.of_trait.and_then(|t| t.trait_def_id())
                && error_def_id == trait_def_id
                && let Some(def_id) = path_res(cx, imp.self_ty).opt_def_id().and_then(DefId::as_local)
                && let hir_id = cx.tcx.hir().local_def_id_to_hir_id(def_id)
                && let Node::Item(ty_item) = cx.tcx.hir().get(hir_id)
                && ty_item.ident.name == sym::Error =>
            {
                span_lint_hir_and_then(
                    cx,
                    ERROR_IMPL_ERROR,
                    hir_id,
                    ty_item.ident.span,
                    "type named `Error` that implements `Error`",
                    |diag| {
                        diag.span_note(item.span, "`Error` was implemented here");
                    }
                );
            }
            _ => {},
        }
        {}
    }
}

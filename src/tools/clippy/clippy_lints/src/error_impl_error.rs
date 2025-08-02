use clippy_utils::diagnostics::{span_lint, span_lint_hir_and_then};
use clippy_utils::path_res;
use clippy_utils::ty::implements_trait;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Visibility;
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for types named `Error` that implement `Error`.
    ///
    /// ### Why restrict this?
    /// It can become confusing when a codebase has 20 types all named `Error`, requiring either
    /// aliasing them in the `use` statement or qualifying them like `my_module::Error`. This
    /// hinders comprehension, as it requires you to memorize every variation of importing `Error`
    /// used across a codebase.
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
    #[clippy::version = "1.73.0"]
    pub ERROR_IMPL_ERROR,
    restriction,
    "exported types named `Error` that implement `Error`"
}
declare_lint_pass!(ErrorImplError => [ERROR_IMPL_ERROR]);

impl<'tcx> LateLintPass<'tcx> for ErrorImplError {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        match item.kind {
            ItemKind::TyAlias(ident, ..)
                if ident.name == sym::Error
                    && is_visible_outside_module(cx, item.owner_id.def_id)
                    && let ty = cx.tcx.type_of(item.owner_id).instantiate_identity()
                    && let Some(error_def_id) = cx.tcx.get_diagnostic_item(sym::Error)
                    && implements_trait(cx, ty, error_def_id, &[]) =>
            {
                span_lint(
                    cx,
                    ERROR_IMPL_ERROR,
                    ident.span,
                    "exported type alias named `Error` that implements `Error`",
                );
            },
            ItemKind::Impl(imp)
                if let Some(trait_def_id) = imp.of_trait.and_then(|t| t.trait_ref.trait_def_id())
                    && let Some(error_def_id) = cx.tcx.get_diagnostic_item(sym::Error)
                    && error_def_id == trait_def_id
                    && let Some(def_id) = path_res(cx, imp.self_ty).opt_def_id().and_then(DefId::as_local)
                    && let Some(ident) = cx.tcx.opt_item_ident(def_id.to_def_id())
                    && ident.name == sym::Error
                    && is_visible_outside_module(cx, def_id) =>
            {
                span_lint_hir_and_then(
                    cx,
                    ERROR_IMPL_ERROR,
                    cx.tcx.local_def_id_to_hir_id(def_id),
                    ident.span,
                    "exported type named `Error` that implements `Error`",
                    |diag| {
                        diag.span_note(item.span, "`Error` was implemented here");
                    },
                );
            },
            _ => {},
        }
    }
}

/// Do not lint private `Error`s, i.e., ones without any `pub` (minus `pub(self)` of course) and
/// which aren't reexported
fn is_visible_outside_module(cx: &LateContext<'_>, def_id: LocalDefId) -> bool {
    !matches!(
        cx.tcx.visibility(def_id),
        Visibility::Restricted(mod_def_id) if cx.tcx.parent_module_from_def_id(def_id).to_def_id() == mod_def_id
    )
}

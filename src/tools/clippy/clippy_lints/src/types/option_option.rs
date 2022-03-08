use clippy_utils::diagnostics::span_lint;
use clippy_utils::{path_def_id, qpath_generic_tys};
use if_chain::if_chain;
use rustc_hir::{self as hir, def_id::DefId, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::OPTION_OPTION;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    if_chain! {
        if cx.tcx.is_diagnostic_item(sym::Option, def_id);
        if let Some(arg) = qpath_generic_tys(qpath).next();
        if path_def_id(cx, arg) == Some(def_id);
        then {
            span_lint(
                cx,
                OPTION_OPTION,
                hir_ty.span,
                "consider using `Option<T>` instead of `Option<Option<T>>` or a custom \
                                     enum if you need to distinguish all 3 cases",
            );
            true
        } else {
            false
        }
    }
}

use rustc_hir::{self as hir, def_id::DefId, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use crate::utils::span_lint;

use super::utils;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    if cx.tcx.is_diagnostic_item(sym::option_type, def_id) {
        if utils::is_ty_param_diagnostic_item(cx, qpath, sym::option_type).is_some() {
            span_lint(
                cx,
                super::OPTION_OPTION,
                hir_ty.span,
                "consider using `Option<T>` instead of `Option<Option<T>>` or a custom \
                                 enum if you need to distinguish all 3 cases",
            );
            return true;
        }
    }
    false
}

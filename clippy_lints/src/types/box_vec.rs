use rustc_hir::{self as hir, def_id::DefId, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use crate::utils::{is_ty_param_diagnostic_item, span_lint_and_help};

use super::BOX_VEC;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    if Some(def_id) == cx.tcx.lang_items().owned_box()
        && is_ty_param_diagnostic_item(cx, qpath, sym::vec_type).is_some()
    {
        span_lint_and_help(
            cx,
            BOX_VEC,
            hir_ty.span,
            "you seem to be trying to use `Box<Vec<T>>`. Consider using just `Vec<T>`",
            None,
            "`Vec<T>` is already on the heap, `Box<Vec<T>>` makes an extra allocation",
        );
        true
    } else {
        false
    }
}

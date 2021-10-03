use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_ty_param_diagnostic_item;
use rustc_hir::{self as hir, def_id::DefId, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::BOX_COLLECTION;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    if_chain! {
        if Some(def_id) == cx.tcx.lang_items().owned_box();
        if let Some(item_type) = get_std_collection(cx, qpath);
        then {
            let generic = if item_type == "String" {
                ""
            } else {
                "<..>"
            };
            span_lint_and_help(
                cx,
                BOX_COLLECTION,
                hir_ty.span,
                &format!(
                    "you seem to be trying to use `Box<{outer}{generic}>`. Consider using just `{outer}{generic}`",
                    outer=item_type,
                    generic = generic),
                None,
                &format!(
                    "`{outer}{generic}` is already on the heap, `Box<{outer}{generic}>` makes an extra allocation",
                    outer=item_type,
                    generic = generic)
            );
            true
        } else {
            false
        }
    }
}

fn get_std_collection(cx: &LateContext<'_>, qpath: &QPath<'_>) -> Option<&'static str> {
    if is_ty_param_diagnostic_item(cx, qpath, sym::Vec).is_some() {
        Some("Vec")
    } else if is_ty_param_diagnostic_item(cx, qpath, sym::String).is_some() {
        Some("String")
    } else if is_ty_param_diagnostic_item(cx, qpath, sym::HashMap).is_some() {
        Some("HashMap")
    } else {
        None
    }
}

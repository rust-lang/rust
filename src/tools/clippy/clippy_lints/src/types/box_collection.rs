use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{path_def_id, qpath_generic_tys};
use rustc_hir::{self as hir, def_id::DefId, QPath};
use rustc_lint::LateContext;
use rustc_span::{sym, Symbol};

use super::BOX_COLLECTION;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    if_chain! {
        if Some(def_id) == cx.tcx.lang_items().owned_box();
        if let Some(item_type) = get_std_collection(cx, qpath);
        then {
            let generic = match item_type {
                sym::String => "",
                _ => "<..>",
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

fn get_std_collection(cx: &LateContext<'_>, qpath: &QPath<'_>) -> Option<Symbol> {
    let param = qpath_generic_tys(qpath).next()?;
    let id = path_def_id(cx, param)?;
    cx.tcx
        .get_diagnostic_name(id)
        .filter(|&name| matches!(name, sym::HashMap | sym::String | sym::Vec))
}

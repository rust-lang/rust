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

            let box_content = format!("{item_type}{generic}");
            span_lint_and_help(
                cx,
                BOX_COLLECTION,
                hir_ty.span,
                &format!(
                    "you seem to be trying to use `Box<{box_content}>`. Consider using just `{box_content}`"),
                None,
                &format!(
                    "`{box_content}` is already on the heap, `Box<{box_content}>` makes an extra allocation")
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
        .filter(|&name| {
            matches!(
                name,
                sym::HashMap
                    | sym::Vec
                    | sym::HashSet
                    | sym::VecDeque
                    | sym::LinkedList
                    | sym::BTreeMap
                    | sym::BTreeSet
                    | sym::BinaryHeap
            )
        })
        .or_else(|| {
            cx.tcx
                .lang_items()
                .string()
                .filter(|did| id == *did)
                .map(|_| sym::String)
        })
}

use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::LINKEDLIST;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, def_id: DefId) -> bool {
    if cx.tcx.is_diagnostic_item(sym::LinkedList, def_id) {
        span_lint_and_help(
            cx,
            LINKEDLIST,
            hir_ty.span,
            "you seem to be using a `LinkedList`! Perhaps you meant some other data structure?",
            None,
            "a `VecDeque` might work",
        );
        true
    } else {
        false
    }
}

use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{match_def_path, paths};
use rustc_hir::{self as hir, def_id::DefId};
use rustc_lint::LateContext;

use super::LINKEDLIST;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, def_id: DefId) -> bool {
    if match_def_path(cx, def_id, &paths::LINKED_LIST) {
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

use clippy_utils::diagnostics::span_lint;
use clippy_utils::res::{MaybeDef as _, MaybeQPath as _};
use clippy_utils::sym;
use rustc_hir::{Expr, Item, ItemKind, OwnerNode};
use rustc_lint::LateContext;

use super::EXIT;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, func: &'tcx Expr<'_>) {
    if func.res(cx).is_diag_item(cx, sym::process_exit)
        && let parent = cx.tcx.hir_get_parent_item(expr.hir_id)
        && let OwnerNode::Item(Item{kind: ItemKind::Fn{ ident, .. }, ..}) = cx.tcx.hir_owner_node(parent)
        // If the next item up is a function we check if it isn't named "main"
        // and only then emit a linter warning

        // if you instead check for the parent of the `exit()` call being the entrypoint function, as this worked before,
        // in compilation contexts like --all-targets (which include --tests), you get false positives
        // because in a test context, main is not the entrypoint function
        && ident.name != sym::main
        && !expr.span.in_external_macro(cx.tcx.sess.source_map())
    {
        span_lint(cx, EXIT, expr.span, "usage of `process::exit`");
    }
}

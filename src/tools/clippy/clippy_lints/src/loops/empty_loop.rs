use super::EMPTY_LOOP;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{is_in_panic_handler, is_no_std_crate, sym};

use rustc_hir::{Block, Expr, ItemKind, Node};
use rustc_lint::LateContext;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, loop_block: &Block<'_>) {
    let parent_hir_id = cx.tcx.parent_hir_id(expr.hir_id);
    if let Node::Item(parent_node) = cx.tcx.hir_node(parent_hir_id)
        && matches!(parent_node.kind, ItemKind::Fn { .. })
        && let attrs = cx.tcx.hir_attrs(parent_hir_id)
        && attrs.iter().any(|attr| attr.has_name(sym::rustc_intrinsic))
    {
        // Intrinsic functions are expanded into an empty loop when lowering the AST
        // to simplify the job of later passes which might expect any function to have a body.
        return;
    }

    if loop_block.stmts.is_empty() && loop_block.expr.is_none() && !is_in_panic_handler(cx, expr) {
        let msg = "empty `loop {}` wastes CPU cycles";
        let help = if is_no_std_crate(cx) {
            "you should either use `panic!()` or add a call pausing or sleeping the thread to the loop body"
        } else {
            "you should either use `panic!()` or add `std::thread::sleep(..);` to the loop body"
        };
        span_lint_and_help(cx, EMPTY_LOOP, expr.span, msg, None, help);
    }
}

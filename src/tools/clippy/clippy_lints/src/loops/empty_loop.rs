use super::EMPTY_LOOP;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{is_in_panic_handler, is_no_std_crate};

use rustc_hir::{Block, Expr};
use rustc_lint::LateContext;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, loop_block: &Block<'_>) {
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

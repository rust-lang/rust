use crate::methods::{single_char_insert_string, single_char_push_string};
use clippy_utils::{match_def_path, paths};
use rustc_hir as hir;
use rustc_lint::LateContext;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, receiver: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    if let Some(fn_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
        if match_def_path(cx, fn_def_id, &paths::PUSH_STR) {
            single_char_push_string::check(cx, expr, receiver, args);
        } else if match_def_path(cx, fn_def_id, &paths::INSERT_STR) {
            single_char_insert_string::check(cx, expr, receiver, args);
        }
    }
}

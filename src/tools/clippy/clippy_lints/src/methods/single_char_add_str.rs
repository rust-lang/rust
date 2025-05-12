use crate::methods::{single_char_insert_string, single_char_push_string};
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, receiver: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    if let Some(fn_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
        if cx.tcx.is_diagnostic_item(sym::string_push_str, fn_def_id) {
            single_char_push_string::check(cx, expr, receiver, args);
        } else if cx.tcx.is_diagnostic_item(sym::string_insert_str, fn_def_id) {
            single_char_insert_string::check(cx, expr, receiver, args);
        }
    }
}

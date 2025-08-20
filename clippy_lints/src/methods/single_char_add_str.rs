use crate::methods::{single_char_insert_string, single_char_push_string};
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, receiver: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    if let Some(fn_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
        match cx.tcx.get_diagnostic_name(fn_def_id) {
            Some(sym::string_push_str) => single_char_push_string::check(cx, expr, receiver, args),
            Some(sym::string_insert_str) => single_char_insert_string::check(cx, expr, receiver, args),
            _ => {},
        }
    }
}

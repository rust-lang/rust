use crate::utils::{
    derefs_to_slice, is_type_diagnostic_item, match_trait_method, method_chain_args, paths, snippet_with_applicability,
    span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::ITER_COUNT;

pub(crate) fn lints<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, iter_args: &'tcx [Expr<'tcx>], is_mut: bool) {
    let mut_str = if is_mut { "_mut" } else { "" };
    let iter_method = if method_chain_args(expr, &[format!("iter{}", mut_str).as_str(), "count"]).is_some() {
        "iter"
    } else if method_chain_args(expr, &["into_iter", "count"]).is_some() {
        "into_iter"
    } else {
        return;
    };
    if_chain! {
        let caller_type = if derefs_to_slice(cx, &iter_args[0], cx.typeck_results().expr_ty(&iter_args[0])).is_some() {
            Some("slice")
        } else if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&iter_args[0]), sym::vec_type) {
            Some("Vec")
        } else if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&iter_args[0]), sym!(vecdeque_type)) {
            Some("VecDeque")
        } else if match_trait_method(cx, expr, &paths::ITERATOR) {
            Some("std::iter::Iterator")
        } else {
            None
        };
        if let Some(caller_type) = caller_type;
        then {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                ITER_COUNT,
                expr.span,
                &format!("called `.{}{}().count()` on a `{}`", iter_method, mut_str, caller_type),
                "try",
                format!(
                    "{}.len()",
                    snippet_with_applicability(cx, iter_args[0].span, "..", &mut applicability),
                ),
                applicability,
            );
        }
    }
}

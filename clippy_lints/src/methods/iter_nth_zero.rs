use crate::consts::{constant, Constant};
use crate::utils::{match_trait_method, paths, snippet_with_applicability, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::ITER_NTH_ZERO;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &hir::Expr<'_>, nth_args: &'tcx [hir::Expr<'_>]) {
    if_chain! {
        if match_trait_method(cx, expr, &paths::ITERATOR);
        if let Some((Constant::Int(0), _)) = constant(cx, cx.typeck_results(), &nth_args[1]);
        then {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                ITER_NTH_ZERO,
                expr.span,
                "called `.nth(0)` on a `std::iter::Iterator`, when `.next()` is equivalent",
                "try calling `.next()` instead of `.nth(0)`",
                format!("{}.next()", snippet_with_applicability(cx, nth_args[0].span, "..", &mut applicability)),
                applicability,
            );
        }
    }
}

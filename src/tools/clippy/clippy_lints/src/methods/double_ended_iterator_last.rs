use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::Instance;
use rustc_span::{Span, sym};

use super::DOUBLE_ENDED_ITERATOR_LAST;

pub(super) fn check(cx: &LateContext<'_>, expr: &'_ Expr<'_>, self_expr: &'_ Expr<'_>, call_span: Span) {
    let typeck = cx.typeck_results();

    // if the "last" method is that of Iterator
    if is_trait_method(cx, expr, sym::Iterator)
        // if self implements DoubleEndedIterator
        && let Some(deiter_id) = cx.tcx.get_diagnostic_item(sym::DoubleEndedIterator)
        && let self_type = cx.typeck_results().expr_ty(self_expr)
        && implements_trait(cx, self_type.peel_refs(), deiter_id, &[])
        // resolve the method definition
        && let id = typeck.type_dependent_def_id(expr.hir_id).unwrap()
        && let args = typeck.node_args(expr.hir_id)
        && let Ok(Some(fn_def)) = Instance::try_resolve(cx.tcx, cx.typing_env(), id, args)
        // find the provided definition of Iterator::last
        && let Some(item) = cx.tcx.get_diagnostic_item(sym::Iterator)
        && let Some(last_def) = cx.tcx.provided_trait_methods(item).find(|m| m.name.as_str() == "last")
        // if the resolved method is the same as the provided definition
        && fn_def.def_id() == last_def.def_id
    {
        span_lint_and_sugg(
            cx,
            DOUBLE_ENDED_ITERATOR_LAST,
            call_span,
            "called `Iterator::last` on a `DoubleEndedIterator`; this will needlessly iterate the entire iterator",
            "try",
            "next_back()".to_string(),
            Applicability::MachineApplicable,
        );
    }
}

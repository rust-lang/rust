use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::Range;
use clippy_utils::is_integer_const;
use rustc_ast::ast::RangeLimits;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use rustc_span::Span;

use super::ITER_WITH_DRAIN;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, span: Span, arg: &Expr<'_>) {
    if !matches!(recv.kind, ExprKind::Field(..))
        && let Some(adt) = cx.typeck_results().expr_ty(recv).ty_adt_def()
        && let Some(ty_name) = cx.tcx.get_diagnostic_name(adt.did())
        && matches!(ty_name, sym::Vec | sym::VecDeque)
        && let Some(range) = Range::hir(arg)
        && is_full_range(cx, recv, range)
    {
        span_lint_and_sugg(
            cx,
            ITER_WITH_DRAIN,
            span.with_hi(expr.span.hi()),
            &format!("`drain(..)` used on a `{}`", ty_name),
            "try this",
            "into_iter()".to_string(),
            Applicability::MaybeIncorrect,
        );
    };
}

fn is_full_range(cx: &LateContext<'_>, container: &Expr<'_>, range: Range<'_>) -> bool {
    range.start.map_or(true, |e| is_integer_const(cx, e, 0))
        && range.end.map_or(true, |e| {
            if range.limits == RangeLimits::HalfOpen
                && let ExprKind::Path(QPath::Resolved(None, container_path)) = container.kind
                && let ExprKind::MethodCall(name, self_arg, [], _) = e.kind
                && name.ident.name == sym::len
                && let ExprKind::Path(QPath::Resolved(None, path)) = self_arg.kind
            {
                container_path.res == path.res
            } else {
                false
            }
        })
}

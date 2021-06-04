use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::ITER_SKIP_NEXT;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>, arg: &hir::Expr<'_>) {
    // lint if caller of skip is an Iterator
    if is_trait_method(cx, expr, sym::Iterator) {
        span_lint_and_sugg(
            cx,
            ITER_SKIP_NEXT,
            expr.span.trim_start(recv.span).unwrap(),
            "called `skip(..).next()` on an iterator",
            "use `nth` instead",
            format!(".nth({})", snippet(cx, arg.span, "..")),
            Applicability::MachineApplicable,
        );
    }
}

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::ITER_SKIP_NEXT;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, skip_args: &[hir::Expr<'_>]) {
    // lint if caller of skip is an Iterator
    if is_trait_method(cx, expr, sym::Iterator) {
        if let [caller, n] = skip_args {
            span_lint_and_sugg(
                cx,
                ITER_SKIP_NEXT,
                expr.span.trim_start(caller.span).unwrap(),
                "called `skip(..).next()` on an iterator",
                "use `nth` instead",
                format!(".nth({})", snippet(cx, n.span, "..")),
                Applicability::MachineApplicable,
            );
        }
    }
}

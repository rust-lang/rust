use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::{expr_or_init, paths};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, path: &Expr<'_>, args: &[Expr<'_>], msrv: Msrv) {
    if let [error_kind, error] = args
        && !expr.span.from_expansion()
        && !error_kind.span.from_expansion()
        && let ExprKind::Path(QPath::TypeRelative(_, new_segment)) = path.kind
        && paths::IO_ERROR_NEW.matches_path(cx, path)
        && paths::IO_ERRORKIND_OTHER_CTOR.matches_path(cx, expr_or_init(cx, error_kind))
        && msrv.meets(cx, msrvs::IO_ERROR_OTHER)
    {
        span_lint_and_then(
            cx,
            super::IO_OTHER_ERROR,
            expr.span,
            "this can be `std::io::Error::other(_)`",
            |diag| {
                diag.multipart_suggestion_verbose(
                    "use `std::io::Error::other`",
                    vec![
                        (new_segment.ident.span, "other".to_owned()),
                        (error_kind.span.until(error.span.source_callsite()), String::new()),
                    ],
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}

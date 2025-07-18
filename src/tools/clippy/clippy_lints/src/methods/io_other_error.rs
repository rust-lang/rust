use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::{expr_or_init, is_path_diagnostic_item, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, path: &Expr<'_>, args: &[Expr<'_>], msrv: Msrv) {
    if let [error_kind, error] = args
        && !expr.span.from_expansion()
        && !error_kind.span.from_expansion()
        && let ExprKind::Path(QPath::TypeRelative(_, new_segment)) = path.kind
        && is_path_diagnostic_item(cx, path, sym::io_error_new)
        && let ExprKind::Path(QPath::Resolved(_, init_path)) = &expr_or_init(cx, error_kind).kind
        && let [.., error_kind_ty, error_kind_variant] = init_path.segments
        && cx.tcx.is_diagnostic_item(sym::io_errorkind, error_kind_ty.res.def_id())
        && error_kind_variant.ident.name == sym::Other
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

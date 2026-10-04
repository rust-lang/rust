use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::MaybeDef as _;
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::Span;

use crate::methods::MAP_UNWRAP_OR;

/// Lints usage of `map(f).unwrap_or_default()`, where the return type of `f` is not `bool`.
/// When `f` returns a `bool` value, the `manual_is_variant_and` lint is more useful.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    map_recv: &'tcx Expr<'tcx>,
    map_method_span: Span,
    msrv: Msrv,
) {
    if !msrv.meets(cx, msrvs::MAP_OR_DEFAULT) {
        return;
    }

    // Consider `Option` and `Result`.
    let Some(ty @ (sym::Option | sym::Result)) =
        cx.typeck_results().expr_ty(map_recv).peel_refs().opt_diag_name(&cx.tcx)
    else {
        return;
    };

    // Ignore `Option<bool>` or `Result<bool>`
    // `manual_is_variant_and` lint is more useful in these cases.
    if cx.typeck_results().expr_ty(expr).is_bool() {
        return;
    }

    // Suggest autofix
    let ty = if ty == sym::Option { "an `Option`" } else { "a `Result`" };
    span_lint_and_then(
        cx,
        MAP_UNWRAP_OR,
        expr.span,
        format!("called `map(_).unwrap_or_default()` on {ty} value"),
        |diag| {
            diag.multipart_suggestion(
                "use instead",
                vec![
                    (recv.span.shrink_to_hi().with_hi(expr.span.hi()), String::new()),
                    (map_method_span, String::from("map_or_default")),
                ],
                Applicability::MachineApplicable,
            );
        },
    );
}

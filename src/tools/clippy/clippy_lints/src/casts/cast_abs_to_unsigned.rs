use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::{meets_msrv, msrvs};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_semver::RustcVersion;

use super::CAST_ABS_TO_UNSIGNED;

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'_>,
    cast_to: Ty<'_>,
    msrv: Option<RustcVersion>,
) {
    if meets_msrv(msrv, msrvs::UNSIGNED_ABS)
        && let ty::Int(from) = cast_from.kind()
        && let ty::Uint(to) = cast_to.kind()
        && let ExprKind::MethodCall(method_path, receiver, ..) = cast_expr.kind
        && method_path.ident.name.as_str() == "abs"
    {
        let span = if from.bit_width() == to.bit_width() {
            expr.span
        } else {
            // if the result of `.unsigned_abs` would be a different type, keep the cast
            // e.g. `i64 -> usize`, `i16 -> u8`
            cast_expr.span
        };

        span_lint_and_sugg(
            cx,
            CAST_ABS_TO_UNSIGNED,
            span,
            &format!("casting the result of `{cast_from}::abs()` to {cast_to}"),
            "replace with",
            format!("{}.unsigned_abs()", Sugg::hir(cx, receiver, "..").maybe_par()),
            Applicability::MachineApplicable,
        );
    }
}

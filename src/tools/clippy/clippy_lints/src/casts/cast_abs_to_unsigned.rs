use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::{meets_msrv, msrvs};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
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
    if_chain! {
        if meets_msrv(msrv, msrvs::UNSIGNED_ABS);
        if cast_from.is_integral();
        if cast_to.is_integral();
        if cast_from.is_signed();
        if !cast_to.is_signed();
        if let ExprKind::MethodCall(method_path, args, _) = cast_expr.kind;
        if let method_name = method_path.ident.name.as_str();
        if method_name == "abs";
        then {
            span_lint_and_sugg(
                cx,
                CAST_ABS_TO_UNSIGNED,
                expr.span,
                &format!("casting the result of `{}::{}()` to {}", cast_from, method_name, cast_to),
                "replace with",
                format!("{}.unsigned_abs()", Sugg::hir(cx, &args[0], "..")),
                Applicability::MachineApplicable,
            );
        }
    }
}

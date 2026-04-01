use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::{is_in_const_context, is_integer_literal, std_or_core};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability, Ty, TyKind};
use rustc_lint::LateContext;

use super::ZERO_PTR;

pub fn check(cx: &LateContext<'_>, expr: &Expr<'_>, from: &Expr<'_>, to: &Ty<'_>, msrv: Msrv) {
    if let TyKind::Ptr(ref mut_ty) = to.kind
        && is_integer_literal(from, 0)
        && (!is_in_const_context(cx) || msrv.meets(cx, msrvs::PTR_NULL))
        && let Some(std_or_core) = std_or_core(cx)
    {
        let (msg, sugg_fn) = match mut_ty.mutbl {
            Mutability::Mut => ("`0 as *mut _` detected", "ptr::null_mut"),
            Mutability::Not => ("`0 as *const _` detected", "ptr::null"),
        };

        let sugg = if let TyKind::Infer(()) = mut_ty.ty.kind {
            format!("{std_or_core}::{sugg_fn}()")
        } else if let Some(mut_ty_snip) = mut_ty.ty.span.get_source_text(cx) {
            format!("{std_or_core}::{sugg_fn}::<{mut_ty_snip}>()")
        } else {
            return;
        };

        span_lint_and_sugg(
            cx,
            ZERO_PTR,
            expr.span,
            msg,
            "try",
            sugg,
            Applicability::MachineApplicable,
        );
    }
}

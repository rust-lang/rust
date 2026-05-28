use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::msrvs::Msrv;
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use clippy_utils::sugg::has_enclosing_paren;
use clippy_utils::{get_parent_expr, is_expr_temporary_value, is_from_proc_macro, is_lint_allowed, msrvs, std_or_core};
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability, Ty, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::{Adjust, AutoBorrow};
use rustc_span::BytePos;

use super::BORROW_AS_PTR;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    cast_expr: &'tcx Expr<'_>,
    cast_to: &'tcx Ty<'_>,
    msrv: Msrv,
) -> bool {
    if let TyKind::Ptr(target) = cast_to.kind
        && !matches!(target.ty.kind, TyKind::TraitObject(..))
        && let ExprKind::AddrOf(BorrowKind::Ref, mutability, e) = cast_expr.kind
        && !is_lint_allowed(cx, BORROW_AS_PTR, expr.hir_id)
        // Fix #9884
        && !is_expr_temporary_value(cx, e)
        && !is_from_proc_macro(cx, expr)
    {
        let mut app = Applicability::MachineApplicable;
        let snip = snippet_with_context(cx, e.span, cast_expr.span.ctxt(), "..", &mut app).0;

        let (suggestion, span) = if msrv.meets(cx, msrvs::RAW_REF_OP) {
            // Make sure that the span to be replaced doesn't include parentheses, that could break the
            // suggestion.
            let span = if has_enclosing_paren(snippet_with_applicability(cx, expr.span, "", &mut app)) {
                expr.span
                    .with_lo(expr.span.lo() + BytePos(1))
                    .with_hi(expr.span.hi() - BytePos(1))
            } else {
                expr.span
            };
            (format!("&raw {} {snip}", mutability.ptr_str()), span)
        } else {
            let Some(std_or_core) = std_or_core(cx) else {
                return false;
            };
            let macro_name = match mutability {
                Mutability::Not => "addr_of",
                Mutability::Mut => "addr_of_mut",
            };
            (format!("{std_or_core}::ptr::{macro_name}!({snip})"), expr.span)
        };

        span_lint_and_sugg(cx, BORROW_AS_PTR, span, "borrow as raw pointer", "try", suggestion, app);
        return true;
    }
    false
}

/// Check for an implicit cast from reference to raw pointer outside an explicit `as`.
pub(super) fn check_implicit_cast(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if !expr.span.from_expansion()
        && let ExprKind::AddrOf(BorrowKind::Ref, _, pointee) = expr.kind
        && !matches!(get_parent_expr(cx, expr).map(|e| e.kind), Some(ExprKind::Cast(..)))
        && let [deref, borrow] = cx.typeck_results().expr_adjustments(expr)
        && matches!(deref.kind, Adjust::Deref(..))
        && let Adjust::Borrow(AutoBorrow::RawPtr(mutability)) = borrow.kind
        // Do not suggest taking a raw pointer to a temporary value
        && !is_expr_temporary_value(cx, pointee)
    {
        span_lint_and_then(cx, BORROW_AS_PTR, expr.span, "implicit borrow as raw pointer", |diag| {
            diag.span_suggestion_verbose(
                expr.span.until(pointee.span),
                "use a raw pointer instead",
                format!("&raw {} ", mutability.ptr_str()),
                Applicability::MachineApplicable,
            );
        });
    }
}

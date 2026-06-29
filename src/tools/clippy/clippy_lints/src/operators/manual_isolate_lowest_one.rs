use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::MaybeResPath;
use clippy_utils::sugg::Sugg;
use clippy_utils::sym;
use rustc_ast::BinOpKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Span;

use super::MANUAL_ISOLATE_LOWEST_ONE;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    op: BinOpKind,
    lhs: &'tcx Expr<'tcx>,
    rhs: &'tcx Expr<'tcx>,
    msrv: Msrv,
) {
    if op != BinOpKind::BitAnd || expr.span.from_expansion() || lhs.span.from_expansion() || rhs.span.from_expansion() {
        return;
    }

    // x & x.wrapping_neg()
    if let Some(base) = matching_wrapping_neg(lhs, rhs)
        // x.wrapping_neg() & x
        .or_else(|| matching_wrapping_neg(rhs, lhs))
        && is_primitive_int(cx, base)
    {
        maybe_emit_lint(cx, expr.span, base, msrv, false);
    } else if let Some(base) = matching_neg(lhs, rhs)
        // x & -x or -x & x
        .or_else(|| matching_neg(rhs, lhs))
        && is_signed_int(cx, base)
    {
        maybe_emit_lint(cx, expr.span, base, msrv, false);
    } else if let Some(base) = matching_nonzero_wrapping_neg(lhs, rhs)
        // nz.get() & nz.get().wrapping_neg() or reversed
        .or_else(|| matching_nonzero_wrapping_neg(rhs, lhs))
        && is_nonzero_int(cx, base)
    {
        maybe_emit_lint(cx, expr.span, base, msrv, true);
    } else if let Some(base) = matching_nonzero_neg(lhs, rhs)
        // nz.get() & -nz.get() or reversed
        .or_else(|| matching_nonzero_neg(rhs, lhs))
        && is_nonzero_signed_int(cx, base)
    {
        maybe_emit_lint(cx, expr.span, base, msrv, true);
    }
}

fn maybe_emit_lint(cx: &LateContext<'_>, span: Span, base: &Expr<'_>, msrv: Msrv, add_get: bool) {
    if !msrv.meets(cx, msrvs::ISOLATE_LOWEST_ONE) {
        return;
    }

    let mut applicability = Applicability::MachineApplicable;
    let snippet = Sugg::hir_with_applicability(cx, base, "_", &mut applicability);
    let get = if add_get { ".get()" } else { "" };

    span_lint_and_sugg(
        cx,
        MANUAL_ISOLATE_LOWEST_ONE,
        span,
        "manual implementation of `isolate_lowest_one`",
        "consider using `.isolate_lowest_one()`",
        format!("{}.isolate_lowest_one(){get}", snippet.maybe_paren()),
        applicability,
    );
}

fn matching_wrapping_neg<'tcx>(base: &'tcx Expr<'tcx>, negated: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    let receiver = wrapping_neg_receiver(negated)?;
    is_same_local_path(base, receiver).then_some(base)
}

fn matching_neg<'tcx>(base: &'tcx Expr<'tcx>, negated: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Unary(UnOp::Neg, inner) = negated.kind
        && is_same_local_path(base, inner)
    {
        Some(base)
    } else {
        None
    }
}

fn matching_nonzero_wrapping_neg<'tcx>(base_get: &'tcx Expr<'tcx>, negated: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    let base = get_receiver(base_get)?;
    let receiver = wrapping_neg_receiver(negated)?;
    let negated_base = get_receiver(receiver)?;
    is_same_local_path(base, negated_base).then_some(base)
}

fn matching_nonzero_neg<'tcx>(base_get: &'tcx Expr<'tcx>, negated: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    let base = get_receiver(base_get)?;
    if let ExprKind::Unary(UnOp::Neg, inner) = negated.kind {
        let negated_base = get_receiver(inner)?;
        is_same_local_path(base, negated_base).then_some(base)
    } else {
        None
    }
}

fn wrapping_neg_receiver<'tcx>(expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::MethodCall(method_name, receiver, [], _) = expr.kind
        && method_name.ident.name == sym::wrapping_neg
    {
        Some(receiver)
    } else {
        None
    }
}

fn get_receiver<'tcx>(expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::MethodCall(method_name, receiver, [], _) = expr.kind
        && method_name.ident.name == sym::get
        && is_local_path(receiver)
    {
        Some(receiver)
    } else {
        None
    }
}

fn is_same_local_path(left: &Expr<'_>, right: &Expr<'_>) -> bool {
    !left.span.from_expansion()
        && !right.span.from_expansion()
        && matches!((left.res_local_id(), right.res_local_id()), (Some(left), Some(right)) if left == right)
}

fn is_local_path(expr: &Expr<'_>) -> bool {
    expr.res_local_id().is_some()
}

fn is_primitive_int(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    matches!(
        cx.typeck_results().expr_ty_adjusted(expr).kind(),
        ty::Int(_) | ty::Uint(_)
    )
}

fn is_signed_int(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    matches!(cx.typeck_results().expr_ty_adjusted(expr).kind(), ty::Int(_))
}

fn is_nonzero_int(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ty::Adt(adt, args) = cx.typeck_results().expr_ty_adjusted(expr).kind()
        && cx.tcx.is_diagnostic_item(sym::NonZero, adt.did())
    {
        matches!(args.type_at(0).kind(), ty::Int(_) | ty::Uint(_))
    } else {
        false
    }
}

fn is_nonzero_signed_int(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ty::Adt(adt, args) = cx.typeck_results().expr_ty_adjusted(expr).kind()
        && cx.tcx.is_diagnostic_item(sym::NonZero, adt.did())
    {
        matches!(args.type_at(0).kind(), ty::Int(_))
    } else {
        false
    }
}

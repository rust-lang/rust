use clippy_utils::consts::{constant_full_int, constant_simple, Constant, FullInt};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{clip, peel_hir_expr_refs, unsext};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, Node};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Span;

use super::IDENTITY_OP;

pub(crate) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
) {
    if !is_allowed(cx, op, left, right) {
        return;
    }

    // we need to know whether a ref is coerced to a value
    // if a ref is coerced, then the suggested lint must deref it
    // e.g. `let _: i32 = x+0` with `x: &i32` should be replaced with `let _: i32 = *x`.
    // we do this by checking the _kind_ of the type of the expression
    // if it's a ref, we then check whether it is erased, and that's it.
    let (peeled_left_span, left_is_coerced_to_value) = {
        let expr = peel_hir_expr_refs(left).0;
        let span = expr.span;
        let is_coerced = expr_is_erased_ref(cx, expr);
        (span, is_coerced)
    };

    let (peeled_right_span, right_is_coerced_to_value) = {
        let expr = peel_hir_expr_refs(right).0;
        let span = expr.span;
        let is_coerced = expr_is_erased_ref(cx, expr);
        (span, is_coerced)
    };

    match op {
        BinOpKind::Add | BinOpKind::BitOr | BinOpKind::BitXor => {
            let _ = check_op(
                cx,
                left,
                0,
                expr.span,
                peeled_right_span,
                needs_parenthesis(cx, expr, right),
                right_is_coerced_to_value,
            ) || check_op(
                cx,
                right,
                0,
                expr.span,
                peeled_left_span,
                Parens::Unneeded,
                left_is_coerced_to_value,
            );
        },
        BinOpKind::Shl | BinOpKind::Shr | BinOpKind::Sub => {
            let _ = check_op(
                cx,
                right,
                0,
                expr.span,
                peeled_left_span,
                Parens::Unneeded,
                left_is_coerced_to_value,
            );
        },
        BinOpKind::Mul => {
            let _ = check_op(
                cx,
                left,
                1,
                expr.span,
                peeled_right_span,
                needs_parenthesis(cx, expr, right),
                right_is_coerced_to_value,
            ) || check_op(
                cx,
                right,
                1,
                expr.span,
                peeled_left_span,
                Parens::Unneeded,
                left_is_coerced_to_value,
            );
        },
        BinOpKind::Div => {
            let _ = check_op(
                cx,
                right,
                1,
                expr.span,
                peeled_left_span,
                Parens::Unneeded,
                left_is_coerced_to_value,
            );
        },
        BinOpKind::BitAnd => {
            let _ = check_op(
                cx,
                left,
                -1,
                expr.span,
                peeled_right_span,
                needs_parenthesis(cx, expr, right),
                right_is_coerced_to_value,
            ) || check_op(
                cx,
                right,
                -1,
                expr.span,
                peeled_left_span,
                Parens::Unneeded,
                left_is_coerced_to_value,
            );
        },
        BinOpKind::Rem => check_remainder(cx, left, right, expr.span, left.span),
        _ => (),
    }
}

fn expr_is_erased_ref(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match cx.typeck_results().expr_ty(expr).kind() {
        ty::Ref(r, ..) => r.is_erased(),
        _ => false,
    }
}

#[derive(Copy, Clone)]
enum Parens {
    Needed,
    Unneeded,
}

/// Checks if `left op right` needs parenthesis when reduced to `right`
/// e.g. `0 + if b { 1 } else { 2 } + if b { 3 } else { 4 }` cannot be reduced
/// to `if b { 1 } else { 2 } + if b { 3 } else { 4 }` where the `if` could be
/// interpreted as a statement
///
/// See #8724
fn needs_parenthesis(cx: &LateContext<'_>, binary: &Expr<'_>, right: &Expr<'_>) -> Parens {
    match right.kind {
        ExprKind::Binary(_, lhs, _) | ExprKind::Cast(lhs, _) => {
            // ensure we're checking against the leftmost expression of `right`
            //
            //     ~~~ `lhs`
            // 0 + {4} * 2
            //     ~~~~~~~ `right`
            return needs_parenthesis(cx, binary, lhs);
        },
        ExprKind::If(..) | ExprKind::Match(..) | ExprKind::Block(..) | ExprKind::Loop(..) => {},
        _ => return Parens::Unneeded,
    }

    let mut prev_id = binary.hir_id;
    for (_, node) in cx.tcx.hir().parent_iter(binary.hir_id) {
        if let Node::Expr(expr) = node
            && let ExprKind::Binary(_, lhs, _) | ExprKind::Cast(lhs, _) = expr.kind
            && lhs.hir_id == prev_id
        {
            // keep going until we find a node that encompasses left of `binary`
            prev_id = expr.hir_id;
            continue;
        }

        match node {
            Node::Block(_) | Node::Stmt(_) => break,
            _ => return Parens::Unneeded,
        };
    }

    Parens::Needed
}

fn is_allowed(cx: &LateContext<'_>, cmp: BinOpKind, left: &Expr<'_>, right: &Expr<'_>) -> bool {
    // This lint applies to integers and their references
    cx.typeck_results().expr_ty(left).peel_refs().is_integral()
        && cx.typeck_results().expr_ty(right).peel_refs().is_integral()
        // `1 << 0` is a common pattern in bit manipulation code
        && !(cmp == BinOpKind::Shl
            && constant_simple(cx, cx.typeck_results(), right) == Some(Constant::Int(0))
            && constant_simple(cx, cx.typeck_results(), left) == Some(Constant::Int(1)))
}

fn check_remainder(cx: &LateContext<'_>, left: &Expr<'_>, right: &Expr<'_>, span: Span, arg: Span) {
    let lhs_const = constant_full_int(cx, cx.typeck_results(), left);
    let rhs_const = constant_full_int(cx, cx.typeck_results(), right);
    if match (lhs_const, rhs_const) {
        (Some(FullInt::S(lv)), Some(FullInt::S(rv))) => lv.abs() < rv.abs(),
        (Some(FullInt::U(lv)), Some(FullInt::U(rv))) => lv < rv,
        _ => return,
    } {
        span_ineffective_operation(cx, span, arg, Parens::Unneeded, false);
    }
}

fn check_op(cx: &LateContext<'_>, e: &Expr<'_>, m: i8, span: Span, arg: Span, parens: Parens, is_erased: bool) -> bool {
    if let Some(Constant::Int(v)) = constant_simple(cx, cx.typeck_results(), e).map(Constant::peel_refs) {
        let check = match *cx.typeck_results().expr_ty(e).peel_refs().kind() {
            ty::Int(ity) => unsext(cx.tcx, -1_i128, ity),
            ty::Uint(uty) => clip(cx.tcx, !0, uty),
            _ => return false,
        };
        if match m {
            0 => v == 0,
            -1 => v == check,
            1 => v == 1,
            _ => unreachable!(),
        } {
            span_ineffective_operation(cx, span, arg, parens, is_erased);
            return true;
        }
    }
    false
}

fn span_ineffective_operation(
    cx: &LateContext<'_>,
    span: Span,
    arg: Span,
    parens: Parens,
    is_ref_coerced_to_val: bool,
) {
    let mut applicability = Applicability::MachineApplicable;
    let expr_snippet = snippet_with_applicability(cx, arg, "..", &mut applicability);
    let expr_snippet = if is_ref_coerced_to_val {
        format!("*{expr_snippet}")
    } else {
        expr_snippet.into_owned()
    };
    let suggestion = match parens {
        Parens::Needed => format!("({expr_snippet})"),
        Parens::Unneeded => expr_snippet,
    };

    span_lint_and_sugg(
        cx,
        IDENTITY_OP,
        span,
        "this operation has no effect",
        "consider reducing it to",
        suggestion,
        applicability,
    );
}

use clippy_utils::consts::{ConstEvalCtxt, Constant, FullInt};
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
            if is_redundant_op(cx, left, 0) {
                let paren = needs_parenthesis(cx, expr, right);
                span_ineffective_operation(cx, expr.span, peeled_right_span, paren, right_is_coerced_to_value);
            } else if is_redundant_op(cx, right, 0) {
                let paren = needs_parenthesis(cx, expr, left);
                span_ineffective_operation(cx, expr.span, peeled_left_span, paren, left_is_coerced_to_value);
            }
        },
        BinOpKind::Shl | BinOpKind::Shr | BinOpKind::Sub => {
            if is_redundant_op(cx, right, 0) {
                let paren = needs_parenthesis(cx, expr, left);
                span_ineffective_operation(cx, expr.span, peeled_left_span, paren, left_is_coerced_to_value);
            }
        },
        BinOpKind::Mul => {
            if is_redundant_op(cx, left, 1) {
                let paren = needs_parenthesis(cx, expr, right);
                span_ineffective_operation(cx, expr.span, peeled_right_span, paren, right_is_coerced_to_value);
            } else if is_redundant_op(cx, right, 1) {
                let paren = needs_parenthesis(cx, expr, left);
                span_ineffective_operation(cx, expr.span, peeled_left_span, paren, left_is_coerced_to_value);
            }
        },
        BinOpKind::Div => {
            if is_redundant_op(cx, right, 1) {
                let paren = needs_parenthesis(cx, expr, left);
                span_ineffective_operation(cx, expr.span, peeled_left_span, paren, left_is_coerced_to_value);
            }
        },
        BinOpKind::BitAnd => {
            if is_redundant_op(cx, left, -1) {
                let paren = needs_parenthesis(cx, expr, right);
                span_ineffective_operation(cx, expr.span, peeled_right_span, paren, right_is_coerced_to_value);
            } else if is_redundant_op(cx, right, -1) {
                let paren = needs_parenthesis(cx, expr, left);
                span_ineffective_operation(cx, expr.span, peeled_left_span, paren, left_is_coerced_to_value);
            }
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

/// Checks if a binary expression needs parenthesis when reduced to just its
/// right or left child.
///
/// e.g. `-(x + y + 0)` cannot be reduced to `-x + y`, as the behavior changes silently.
/// e.g. `1u64 + ((x + y + 0i32) as u64)` cannot be reduced to `1u64 + x + y as u64`, since
/// the cast expression will not apply to the same expression.
/// e.g. `0 + if b { 1 } else { 2 } + if b { 3 } else { 4 }` cannot be reduced
/// to `if b { 1 } else { 2 } + if b { 3 } else { 4 }` where the `if` could be
/// interpreted as a statement. The same behavior happens for `match`, `loop`,
/// and blocks.
/// e.g.  `2 * (0 + { a })` can be reduced to `2 * { a }` without the need for parenthesis,
/// but `1 * ({ a } + 4)` cannot be reduced to `{ a } + 4`, as a block at the start of a line
/// will be interpreted as a statement instead of an expression.
///
/// See #8724, #13470
fn needs_parenthesis(cx: &LateContext<'_>, binary: &Expr<'_>, child: &Expr<'_>) -> Parens {
    match child.kind {
        ExprKind::Binary(_, lhs, _) | ExprKind::Cast(lhs, _) => {
            // For casts and binary expressions, we want to add parenthesis if
            // the parent HIR node is an expression, or if the parent HIR node
            // is a Block or Stmt, and the new left hand side would need
            // parenthesis be treated as a statement rather than an expression.
            if let Some((_, parent)) = cx.tcx.hir_parent_iter(binary.hir_id).next() {
                match parent {
                    Node::Expr(_) => return Parens::Needed,
                    Node::Block(_) | Node::Stmt(_) => {
                        // ensure we're checking against the leftmost expression of `child`
                        //
                        // ~~~~~~~~~~~ `binary`
                        //     ~~~ `lhs`
                        // 0 + {4} * 2
                        //     ~~~~~~~ `child`
                        return needs_parenthesis(cx, binary, lhs);
                    },
                    _ => return Parens::Unneeded,
                }
            }
        },
        ExprKind::If(..) | ExprKind::Match(..) | ExprKind::Block(..) | ExprKind::Loop(..) => {
            // For if, match, block, and loop expressions, we want to add parenthesis if
            // the closest ancestor node that is not an expression is a block or statement.
            // This would mean that the rustfix suggestion will appear at the start of a line, which causes
            // these expressions to be interpreted as statements if they do not have parenthesis.
            let mut prev_id = binary.hir_id;
            for (_, parent) in cx.tcx.hir_parent_iter(binary.hir_id) {
                if let Node::Expr(expr) = parent
                    && let ExprKind::Binary(_, lhs, _) | ExprKind::Cast(lhs, _) | ExprKind::Unary(_, lhs) = expr.kind
                    && lhs.hir_id == prev_id
                {
                    // keep going until we find a node that encompasses left of `binary`
                    prev_id = expr.hir_id;
                    continue;
                }

                match parent {
                    Node::Block(_) | Node::Stmt(_) => return Parens::Needed,
                    _ => return Parens::Unneeded,
                };
            }
        },
        _ => {
            return Parens::Unneeded;
        },
    }
    Parens::Needed
}

fn is_allowed(cx: &LateContext<'_>, cmp: BinOpKind, left: &Expr<'_>, right: &Expr<'_>) -> bool {
    // This lint applies to integers and their references
    cx.typeck_results().expr_ty(left).peel_refs().is_integral()
        && cx.typeck_results().expr_ty(right).peel_refs().is_integral()
        // `1 << 0` is a common pattern in bit manipulation code
        && !(cmp == BinOpKind::Shl
            && ConstEvalCtxt::new(cx).eval_simple(right) == Some(Constant::Int(0))
            && ConstEvalCtxt::new(cx).eval_simple(left) == Some(Constant::Int(1)))
}

fn check_remainder(cx: &LateContext<'_>, left: &Expr<'_>, right: &Expr<'_>, span: Span, arg: Span) {
    let ecx = ConstEvalCtxt::new(cx);
    if match (ecx.eval_full_int(left), ecx.eval_full_int(right)) {
        (Some(FullInt::S(lv)), Some(FullInt::S(rv))) => lv.abs() < rv.abs(),
        (Some(FullInt::U(lv)), Some(FullInt::U(rv))) => lv < rv,
        _ => return,
    } {
        span_ineffective_operation(cx, span, arg, Parens::Unneeded, false);
    }
}

fn is_redundant_op(cx: &LateContext<'_>, e: &Expr<'_>, m: i8) -> bool {
    if let Some(Constant::Int(v)) = ConstEvalCtxt::new(cx).eval_simple(e).map(Constant::peel_refs) {
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

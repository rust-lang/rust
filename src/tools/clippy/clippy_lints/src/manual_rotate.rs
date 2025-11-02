use std::fmt::Display;

use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// It detects manual bit rotations that could be rewritten using standard
    /// functions `rotate_left` or `rotate_right`.
    ///
    /// ### Why is this bad?
    ///
    /// Calling the function better conveys the intent.
    ///
    /// ### Known issues
    ///
    /// Currently, the lint only catches shifts by constant amount.
    ///
    /// ### Example
    /// ```no_run
    /// let x = 12345678_u32;
    /// let _ = (x >> 8) | (x << 24);
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x = 12345678_u32;
    /// let _ = x.rotate_right(8);
    /// ```
    #[clippy::version = "1.81.0"]
    pub MANUAL_ROTATE,
    style,
    "using bit shifts to rotate integers"
}

declare_lint_pass!(ManualRotate => [MANUAL_ROTATE]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShiftDirection {
    Left,
    Right,
}

impl Display for ShiftDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Left => "rotate_left",
            Self::Right => "rotate_right",
        })
    }
}

fn parse_shift<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<(ShiftDirection, &'tcx Expr<'tcx>, &'tcx Expr<'tcx>)> {
    if let ExprKind::Binary(op, l, r) = expr.kind {
        let dir = match op.node {
            BinOpKind::Shl => ShiftDirection::Left,
            BinOpKind::Shr => ShiftDirection::Right,
            _ => return None,
        };
        return Some((dir, l, r));
    }
    None
}

impl LateLintPass<'_> for ManualRotate {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let ExprKind::Binary(op, l, r) = expr.kind
            && let BinOpKind::Add | BinOpKind::BitOr = op.node
            && let Some((l_shift_dir, l_expr, l_amount)) = parse_shift(l)
            && let Some((r_shift_dir, r_expr, r_amount)) = parse_shift(r)
            && l_shift_dir != r_shift_dir
            && clippy_utils::eq_expr_value(cx, l_expr, r_expr)
            && let Some(bit_width) = match cx.typeck_results().expr_ty(expr).kind() {
                ty::Int(itype) => itype.bit_width(),
                ty::Uint(itype) => itype.bit_width(),
                _ => return,
            }
        {
            let const_eval = ConstEvalCtxt::new(cx);

            let ctxt = expr.span.ctxt();
            let (shift_function, amount) = if let Some(Constant::Int(l_amount_val)) =
                const_eval.eval_local(l_amount, ctxt)
                && let Some(Constant::Int(r_amount_val)) = const_eval.eval_local(r_amount, ctxt)
                && l_amount_val + r_amount_val == u128::from(bit_width)
            {
                if l_amount_val < r_amount_val {
                    (l_shift_dir, l_amount)
                } else {
                    (r_shift_dir, r_amount)
                }
            } else {
                let (amount1, binop, minuend, amount2, shift_direction) = match (l_amount.kind, r_amount.kind) {
                    (_, ExprKind::Binary(binop, minuend, other)) => (l_amount, binop, minuend, other, l_shift_dir),
                    (ExprKind::Binary(binop, minuend, other), _) => (r_amount, binop, minuend, other, r_shift_dir),
                    _ => return,
                };

                if let Some(Constant::Int(minuend)) = const_eval.eval_local(minuend, ctxt)
                    && clippy_utils::eq_expr_value(cx, amount1, amount2)
                    // (x << s) | (x >> bit_width - s)
                    && ((binop.node == BinOpKind::Sub && u128::from(bit_width) == minuend)
                        // (x << s) | (x >> (bit_width - 1) ^ s)
                        || (binop.node == BinOpKind::BitXor && u128::from(bit_width).checked_sub(minuend) == Some(1)))
                {
                    // NOTE: we take these from the side that _doesn't_ have the binop, since it's probably simpler
                    (shift_direction, amount1)
                } else {
                    return;
                }
            };

            let mut applicability = Applicability::MachineApplicable;
            let expr_sugg = sugg::Sugg::hir_with_applicability(cx, l_expr, "_", &mut applicability).maybe_paren();
            let amount = sugg::Sugg::hir_with_applicability(cx, amount, "_", &mut applicability);
            span_lint_and_sugg(
                cx,
                MANUAL_ROTATE,
                expr.span,
                "there is no need to manually implement bit rotation",
                "this expression can be rewritten as",
                format!("{expr_sugg}.{shift_function}({amount})"),
                Applicability::MachineApplicable,
            );
        }
    }
}

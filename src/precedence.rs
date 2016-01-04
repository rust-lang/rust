use rustc::lint::*;
use syntax::codemap::Spanned;
use syntax::ast::*;

use utils::{span_lint, snippet};

/// **What it does:** This lint checks for operations where precedence may be unclear and `Warn`s about them by default, suggesting to add parentheses. Currently it catches the following:
/// * mixed usage of arithmetic and bit shifting/combining operators without parentheses
/// * a "negative" numeric literal (which is really a unary `-` followed by a numeric literal) followed by a method call
///
/// **Why is this bad?** Because not everyone knows the precedence of those operators by heart, so expressions like these may trip others trying to reason about the code.
///
/// **Known problems:** None
///
/// **Examples:**
/// * `1 << 2 + 3` equals 32, while `(1 << 2) + 3` equals 7
/// * `-1i32.abs()` equals -1, while `(-1i32).abs()` equals 1
declare_lint!(pub PRECEDENCE, Warn,
              "catches operations where precedence may be unclear. See the wiki for a \
               list of cases caught");

#[derive(Copy,Clone)]
pub struct Precedence;

impl LintPass for Precedence {
    fn get_lints(&self) -> LintArray {
        lint_array!(PRECEDENCE)
    }
}

impl EarlyLintPass for Precedence {
    fn check_expr(&mut self, cx: &EarlyContext, expr: &Expr) {
        if let ExprBinary(Spanned { node: op, ..}, ref left, ref right) = expr.node {
            if !is_bit_op(op) {
                return;
            }
            match (is_arith_expr(left), is_arith_expr(right)) {
                (true, true) => {
                    span_lint(cx,
                              PRECEDENCE,
                              expr.span,
                              &format!("operator precedence can trip the unwary. Consider parenthesizing your \
                                        expression:`({}) {} ({})`",
                                       snippet(cx, left.span, ".."),
                                       op.to_string(),
                                       snippet(cx, right.span, "..")));
                }
                (true, false) => {
                    span_lint(cx,
                              PRECEDENCE,
                              expr.span,
                              &format!("operator precedence can trip the unwary. Consider parenthesizing your \
                                        expression:`({}) {} {}`",
                                       snippet(cx, left.span, ".."),
                                       op.to_string(),
                                       snippet(cx, right.span, "..")));
                }
                (false, true) => {
                    span_lint(cx,
                              PRECEDENCE,
                              expr.span,
                              &format!("operator precedence can trip the unwary. Consider parenthesizing your \
                                        expression:`{} {} ({})`",
                                       snippet(cx, left.span, ".."),
                                       op.to_string(),
                                       snippet(cx, right.span, "..")));
                }
                _ => (),
            }
        }

        if let ExprUnary(UnNeg, ref rhs) = expr.node {
            if let ExprMethodCall(_, _, ref args) = rhs.node {
                if let Some(slf) = args.first() {
                    if let ExprLit(ref lit) = slf.node {
                        match lit.node {
                            LitInt(..) | LitFloat(..) | LitFloatUnsuffixed(..) => {
                                span_lint(cx,
                                          PRECEDENCE,
                                          expr.span,
                                          &format!("unary minus has lower precedence than method call. Consider \
                                                    adding parentheses to clarify your intent: -({})",
                                                   snippet(cx, rhs.span, "..")));
                            }
                            _ => (),
                        }
                    }
                }
            }
        }
    }
}

fn is_arith_expr(expr: &Expr) -> bool {
    match expr.node {
        ExprBinary(Spanned { node: op, ..}, _, _) => is_arith_op(op),
        _ => false,
    }
}

fn is_bit_op(op: BinOp_) -> bool {
    match op {
        BiBitXor | BiBitAnd | BiBitOr | BiShl | BiShr => true,
        _ => false,
    }
}

fn is_arith_op(op: BinOp_) -> bool {
    match op {
        BiAdd | BiSub | BiMul | BiDiv | BiRem => true,
        _ => false,
    }
}

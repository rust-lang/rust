use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::Spanned;

use utils::span_lint;

declare_lint!(pub PRECEDENCE, Warn,
              "catches operations where precedence may be unclear. See the wiki for a \
               list of cases caught");

#[derive(Copy,Clone)]
pub struct Precedence;

impl LintPass for Precedence {
    fn get_lints(&self) -> LintArray {
        lint_array!(PRECEDENCE)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprBinary(Spanned { node: op, ..}, ref left, ref right) = expr.node {
            if is_bit_op(op) && (is_arith_expr(left) || is_arith_expr(right)) {
                span_lint(cx, PRECEDENCE, expr.span,
                    "operator precedence can trip the unwary. Consider adding parentheses \
                     to the subexpression");
            }
        }

        if let ExprUnary(UnNeg, ref rhs) = expr.node {
            if let ExprMethodCall(_, _, ref args) = rhs.node {
                if let Some(slf) = args.first() {
                    if let ExprLit(ref lit) = slf.node {
                        match lit.node {
                            LitInt(..) | LitFloat(..) | LitFloatUnsuffixed(..) =>
                                span_lint(cx, PRECEDENCE, expr.span,
                                    "unary minus has lower precedence than method call. Consider \
                                     adding parentheses to clarify your intent"),
                                _ => ()
                        }
                    }
                }
            }
        }
    }
}

fn is_arith_expr(expr : &Expr) -> bool {
    match expr.node {
        ExprBinary(Spanned { node: op, ..}, _, _) => is_arith_op(op),
        _ => false
    }
}

fn is_bit_op(op : BinOp_) -> bool {
    match op {
        BiBitXor | BiBitAnd | BiBitOr | BiShl | BiShr => true,
        _ => false
    }
}

fn is_arith_op(op : BinOp_) -> bool {
    match op {
        BiAdd | BiSub | BiMul | BiDiv | BiRem => true,
        _ => false
    }
}

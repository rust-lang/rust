use consts::{constant_simple, Constant};
use rustc::lint::*;
use rustc::hir::*;
use syntax::codemap::Span;
use utils::{span_lint, snippet, in_macro};
use rustc_const_math::ConstInt;

/// **What it does:** Checks for identity operations, e.g. `x + 0`.
///
/// **Why is this bad?** This code can be removed without changing the
/// meaning. So it just obscures what's going on. Delete it mercilessly.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// x / 1 + 0 * 1 - 0 | 0
/// ```
declare_lint! {
    pub IDENTITY_OP, Warn,
    "using identity operations, e.g. `x + 0` or `y / 1`"
}

#[derive(Copy,Clone)]
pub struct IdentityOp;

impl LintPass for IdentityOp {
    fn get_lints(&self) -> LintArray {
        lint_array!(IDENTITY_OP)
    }
}

impl LateLintPass for IdentityOp {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if in_macro(cx, e.span) {
            return;
        }
        if let ExprBinary(ref cmp, ref left, ref right) = e.node {
            match cmp.node {
                BiAdd | BiBitOr | BiBitXor => {
                    check(cx, left, 0, e.span, right.span);
                    check(cx, right, 0, e.span, left.span);
                }
                BiShl | BiShr | BiSub => check(cx, right, 0, e.span, left.span),
                BiMul => {
                    check(cx, left, 1, e.span, right.span);
                    check(cx, right, 1, e.span, left.span);
                }
                BiDiv => check(cx, right, 1, e.span, left.span),
                BiBitAnd => {
                    check(cx, left, -1, e.span, right.span);
                    check(cx, right, -1, e.span, left.span);
                }
                _ => (),
            }
        }
    }
}


fn check(cx: &LateContext, e: &Expr, m: i8, span: Span, arg: Span) {
    if let Some(v @ Constant::Int(_)) = constant_simple(e) {
        if match m {
            0 => v == Constant::Int(ConstInt::Infer(0)),
            -1 => v == Constant::Int(ConstInt::InferSigned(-1)),
            1 => v == Constant::Int(ConstInt::Infer(1)),
            _ => unreachable!(),
        } {
            span_lint(cx,
                      IDENTITY_OP,
                      span,
                      &format!("the operation is ineffective. Consider reducing it to `{}`",
                               snippet(cx, arg, "..")));
        }
    }
}

use rustc::lint::*;
use rustc::middle::const_eval::lookup_const_by_id;
use rustc::middle::def::*;
use syntax::ast::*;
use syntax::codemap::Span;

use consts::{constant, Constant, is_negative};
use consts::ConstantVariant::ConstantInt;
use utils::{span_lint, snippet};

declare_lint! { pub IDENTITY_OP, Warn,
                "using identity operations, e.g. `x + 0` or `y / 1`" }

#[derive(Copy,Clone)]
pub struct IdentityOp;

impl LintPass for IdentityOp {
    fn get_lints(&self) -> LintArray {
        lint_array!(IDENTITY_OP)
    }

    fn check_expr(&mut self, cx: &Context, e: &Expr) {
        if let ExprBinary(ref cmp, ref left, ref right) = e.node {
            match cmp.node {
                BiAdd | BiBitOr | BiBitXor => {
                    check(cx, left, 0, e.span, right.span);
                    check(cx, right, 0, e.span, left.span);
                },
                BiShl | BiShr | BiSub =>
                    check(cx, right, 0, e.span, left.span),
                BiMul => {
                    check(cx, left, 1, e.span, right.span);
                    check(cx, right, 1, e.span, left.span);
                },
                BiDiv =>
                    check(cx, right, 1, e.span, left.span),
                BiBitAnd => {
                    check(cx, left, -1, e.span, right.span);
                    check(cx, right, -1, e.span, left.span);
                },
                _ => ()
            }
        }
    }
}


fn check(cx: &Context, e: &Expr, m: i8, span: Span, arg: Span) {
    if let Some(c) = constant(cx, e) {
        if c.needed_resolution { return; } // skip linting w/ lookup for now
        if let ConstantInt(v, ty) = c.constant {
            if match m {
                0 => v == 0,
                -1 => is_negative(ty),
                1 => !is_negative(ty),
                _ => unreachable!(),
            } {
                span_lint(cx, IDENTITY_OP, span, &format!(
                    "the operation is ineffective. Consider reducing it to `{}`",
                   snippet(cx, arg, "..")));
            }
        }
    }
}

use rustc::lint::*;
use rustc_front::hir::*;
use syntax::codemap::Span;

use consts::{constant, is_negative};
use consts::Constant::ConstantInt;
use utils::{span_lint, snippet, in_external_macro};

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
    if let Some((c, needed_resolution)) = constant(cx, e) {
        if needed_resolution { return; } // skip linting w/ lookup for now
        if let ConstantInt(v, ty) = c {
            if match m {
                0 => v == 0,
                -1 => is_negative(ty) && v == 1,
                1 => !is_negative(ty) && v == 1,
                _ => unreachable!(),
            } {
                if in_external_macro(cx, e.span) {return;}
                span_lint(cx, IDENTITY_OP, span, &format!(
                    "the operation is ineffective. Consider reducing it to `{}`",
                   snippet(cx, arg, "..")));
            }
        }
    }
}

use rustc::plugin::Registry;
use rustc::lint::*;
use rustc::middle::const_eval::lookup_const_by_id;
use rustc::middle::def::*;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use syntax::ptr::P;
use syntax::codemap::Span;

use utils::{span_lint, snippet};

declare_lint! { pub IDENTITY_OP, Warn,
    "Warn on identity operations, e.g. '_ + 0'"}
    
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
    if have_lit(cx, e, m) {
        span_lint(cx, IDENTITY_OP, span, &format!(
            "The operation is ineffective. Consider reducing it to '{}'", 
           snippet(cx, arg, "..")));
    }
}

fn have_lit(cx: &Context, e : &Expr, m: i8) -> bool {
    match &e.node {
        &ExprUnary(UnNeg, ref litexp) => have_lit(cx, litexp, -m), 
        &ExprLit(ref lit) => {
            match (&lit.node, m) {
                (&LitInt(0, _), 0) => true,
                (&LitInt(1, SignedIntLit(_, Plus)), 1) => true,
                (&LitInt(1, UnsuffixedIntLit(Plus)), 1) => true,
                (&LitInt(1, SignedIntLit(_, Minus)), -1) => true,
                (&LitInt(1, UnsuffixedIntLit(Minus)), -1) => true,
                _ => false
            }
        },
        &ExprParen(ref p) => have_lit(cx, p, m),
        &ExprPath(_, _) => { 
            match cx.tcx.def_map.borrow().get(&e.id) {
                Some(&PathResolution { base_def: DefConst(id), ..}) => 
                        lookup_const_by_id(cx.tcx, id, Option::None)
                        .map_or(false, |l| have_lit(cx, l, m)),
                _ => false
            }
        },
        _ => false
    }
}

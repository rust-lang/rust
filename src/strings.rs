//! This LintPass catches both string addition and string addition + assignment
//!
//! Note that since we have two lints where one subsumes the other, we try to
//! disable the subsumed lint unless it has a higher level

use rustc::lint::*;
use rustc::middle::ty::TypeVariants::TyStruct;
use syntax::ast::*;
use syntax::codemap::{Span, Spanned};
use eq_op::is_exp_equal;
use types::match_ty_unwrap;
use utils::{match_def_path, span_lint, walk_ptrs_ty};

declare_lint! {
    pub STRING_ADD_ASSIGN,
    Warn,
    "Warn on `x = x + ..` where x is a `String`"
}

#[derive(Copy,Clone)]
pub struct StringAdd;

impl LintPass for StringAdd {
    fn get_lints(&self) -> LintArray {
        lint_array!(STRING_ADD_ASSIGN)
    }

    fn check_expr(&mut self, cx: &Context, e: &Expr) {
        if let &ExprAssign(ref target, ref  src) = &e.node {
            if is_string(cx, target) && is_add(src, target) {
                span_lint(cx, STRING_ADD_ASSIGN, e.span,
                    "you assign the result of adding something to this string. \
                    Consider using `String::push_str()` instead.")
            }
        }
    }
}

fn is_string(cx: &Context, e: &Expr) -> bool {
    if let TyStruct(did, _) = walk_ptrs_ty(cx.tcx.expr_ty(e)).sty {
        match_def_path(cx, did.did, &["std", "string", "String"])
    } else { false }
}

fn is_add(src: &Expr, target: &Expr) -> bool {
    match &src.node {
        &ExprBinary(Spanned{ node: BiAdd, .. }, ref left, _) =>
            is_exp_equal(target, left),
        &ExprBlock(ref block) => block.stmts.is_empty() &&
            block.expr.as_ref().map_or(false, |expr| is_add(&*expr, target)),
        &ExprParen(ref expr) => is_add(&*expr, target),
        _ => false
    }
}

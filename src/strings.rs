//! This LintPass catches both string addition and string addition + assignment
//!
//! Note that since we have two lints where one subsumes the other, we try to
//! disable the subsumed lint unless it has a higher level

use rustc::lint::*;
use rustc::middle::ty::TypeVariants::TyStruct;
use syntax::ast::*;
use syntax::codemap::Spanned;

use eq_op::is_exp_equal;
use utils::{match_def_path, span_lint, walk_ptrs_ty, get_parent_expr};

declare_lint! {
    pub STRING_ADD_ASSIGN,
    Allow,
    "using `x = x + ..` where x is a `String`; suggests using `push_str()` instead"
}

declare_lint! {
    pub STRING_ADD,
    Allow,
    "using `x + ..` where x is a `String`; suggests using `push_str()` instead"
}

#[derive(Copy, Clone)]
pub struct StringAdd;

impl LintPass for StringAdd {
    fn get_lints(&self) -> LintArray {
        lint_array!(STRING_ADD, STRING_ADD_ASSIGN)
    }

    fn check_expr(&mut self, cx: &Context, e: &Expr) {
        if let &ExprBinary(Spanned{ node: BiAdd, .. }, ref left, _) = &e.node {
            if is_string(cx, left) {
                if let Allow = cx.current_level(STRING_ADD_ASSIGN) {
                    // the string_add_assign is allow, so no duplicates
                } else {
                    let parent = get_parent_expr(cx, e);
                    if let Some(ref p) = parent {
                        if let &ExprAssign(ref target, _) = &p.node {
                            // avoid duplicate matches
                            if is_exp_equal(cx, target, left) { return; }
                        }
                    }
                }
                //TODO check for duplicates
                 span_lint(cx, STRING_ADD, e.span,
                        "you added something to a string. \
                         Consider using `String::push_str()` instead")
            }
        } else if let &ExprAssign(ref target, ref  src) = &e.node {
            if is_string(cx, target) && is_add(cx, src, target) {
                span_lint(cx, STRING_ADD_ASSIGN, e.span,
                    "you assigned the result of adding something to this string. \
                     Consider using `String::push_str()` instead")
            }
        }
    }
}

fn is_string(cx: &Context, e: &Expr) -> bool {
    let ty = walk_ptrs_ty(cx.tcx.expr_ty(e));
    if let TyStruct(did, _) = ty.sty {
        match_def_path(cx, did.did, &["collections", "string", "String"])
    } else { false }
}

fn is_add(cx: &Context, src: &Expr, target: &Expr) -> bool {
    match &src.node {
        &ExprBinary(Spanned{ node: BiAdd, .. }, ref left, _) =>
            is_exp_equal(cx, target, left),
        &ExprBlock(ref block) => block.stmts.is_empty() &&
            block.expr.as_ref().map_or(false,
                |expr| is_add(cx, &*expr, target)),
        &ExprParen(ref expr) => is_add(cx, &*expr, target),
        _ => false
    }
}

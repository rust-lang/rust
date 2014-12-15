use syntax::ptr::P;
use syntax::ast;
use syntax::ast::*;
use rustc::lint::{Context, LintPass, LintArray, Lint, Level};
use rustc::middle::ty::{mod, expr_ty, ty_str, ty_ptr, ty_rptr};
use syntax::codemap::Span;

use types::span_note_and_lint;

/// Handles uncategorized lints
/// Currently handles linting of if-let-able matches
pub struct MiscPass;


declare_lint!(CLIPPY_SINGLE_MATCH, Warn,
              "Warn on usage of matches with a single nontrivial arm")

impl LintPass for MiscPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CLIPPY_SINGLE_MATCH)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprMatch(ref ex, ref arms, MatchNormal) = expr.node {
            if arms.len() == 2 {
                if arms[0].guard.is_none() && arms[1].pats.len() == 1 {
                    match arms[1].body.node {
                        ExprTup(ref v) if v.len() == 0 && arms[1].guard.is_none() => (),
                        ExprBlock(ref b) if b.stmts.len() == 0 && arms[1].guard.is_none() => (),
                         _ => return
                    }
                    // In some cases, an exhaustive match is preferred to catch situations when
                    // an enum is extended. So we only consider cases where a `_` wildcard is used
                    if arms[1].pats[0].node == PatWild(PatWildSingle) && arms[0].pats.len() == 1 {
                        let map = cx.sess().codemap();
                        span_note_and_lint(cx, CLIPPY_SINGLE_MATCH, expr.span,
                              "You seem to be trying to use match for destructuring a single type. Did you mean to use `if let`?",
                              format!("Try if let {} = {} {{ ... }}",
                                      map.span_to_snippet(arms[0].pats[0].span).unwrap_or("..".to_string()),
                                      map.span_to_snippet(ex.span).unwrap_or("..".to_string())).as_slice()
                        );                        
                    }
                }
            }
        }
    }
}


declare_lint!(CLIPPY_STR_TO_STRING, Warn, "Warn when a String could use into_string() instead of to_string()")

pub struct StrToStringPass;

impl LintPass for StrToStringPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CLIPPY_STR_TO_STRING)
    }

    fn check_expr(&mut self, cx: &Context, expr: &ast::Expr) {
        match expr.node {
            ast::ExprMethodCall(ref method, _, ref args)
                if method.node.as_str() == "to_string"
                && is_str(cx, &*args[0]) => {
                cx.span_lint(CLIPPY_STR_TO_STRING, expr.span, "str.into_string() is faster");
            },
            _ => ()
        }

        fn is_str(cx: &Context, expr: &ast::Expr) -> bool {
            fn walk_ty<'t>(ty: ty::Ty<'t>) -> ty::Ty<'t> {
                //println!("{}: -> {}", depth, ty);
                match ty.sty {
                    ty_ptr(ref tm) | ty_rptr(_, ref tm) => walk_ty(tm.ty),
                    _ => ty
                }
            }
            match walk_ty(expr_ty(cx.tcx, expr)).sty {
                ty_str => true,
                _ => false
            }
        }
    }
}

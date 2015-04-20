use syntax::ptr::P;
use syntax::ast;
use syntax::ast::*;
use syntax::visit::{FnKind};
use rustc::lint::{Context, LintPass, LintArray, Lint, Level};
use rustc::middle::ty::{self, expr_ty, ty_str, ty_ptr, ty_rptr};
use syntax::codemap::Span;

use types::span_note_and_lint;

/// Handles uncategorized lints
/// Currently handles linting of if-let-able matches
#[allow(missing_copy_implementations)]
pub struct MiscPass;


declare_lint!(pub SINGLE_MATCH, Warn,
              "Warn on usage of matches with a single nontrivial arm");

impl LintPass for MiscPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SINGLE_MATCH)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprMatch(ref ex, ref arms, ast::MatchSource::Normal) = expr.node {
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
                        span_note_and_lint(cx, SINGLE_MATCH, expr.span,
                              "You seem to be trying to use match for destructuring a single type. Did you mean to use `if let`?",
                              &*format!("Try if let {} = {} {{ ... }}",
                                      &*map.span_to_snippet(arms[0].pats[0].span).unwrap_or("..".to_string()),
                                      &*map.span_to_snippet(ex.span).unwrap_or("..".to_string()))
                        );
                    }
                }
            }
        }
    }
}


declare_lint!(pub STR_TO_STRING, Warn, "Warn when a String could use to_owned() instead of to_string()");

#[allow(missing_copy_implementations)]
pub struct StrToStringPass;

impl LintPass for StrToStringPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(STR_TO_STRING)
    }

    fn check_expr(&mut self, cx: &Context, expr: &ast::Expr) {
        match expr.node {
            ast::ExprMethodCall(ref method, _, ref args)
                if method.node.as_str() == "to_string"
                && is_str(cx, &*args[0]) => {
                cx.span_lint(STR_TO_STRING, expr.span, "str.to_owned() is faster");
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


declare_lint!(pub TOPLEVEL_REF_ARG, Warn, "Warn about pattern matches with top-level `ref` bindings");

#[allow(missing_copy_implementations)]
pub struct TopLevelRefPass;

impl LintPass for TopLevelRefPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TOPLEVEL_REF_ARG)
    }

    fn check_fn(&mut self, cx: &Context, _: FnKind, decl: &FnDecl, _: &Block, _: Span, _: NodeId) {
        for ref arg in decl.inputs.iter() {
            if let PatIdent(BindByRef(_), _, _) = arg.pat.node {
                cx.span_lint(
                    TOPLEVEL_REF_ARG,
                    arg.pat.span,
                    "`ref` directly on a function argument is ignored. Have you considered using a reference type instead?"
                );
            }
        }
    }
}

use syntax::ptr::P;
use syntax::ast;
use syntax::ast::*;
use rustc::lint::{Context, LintPass, LintArray, Lint, Level};
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

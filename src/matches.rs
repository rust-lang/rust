use rustc::lint::*;
use syntax::ast;
use syntax::ast::*;

use utils::{snippet, span_lint, span_help_and_lint, in_external_macro, expr_block};

declare_lint!(pub SINGLE_MATCH, Warn,
              "a match statement with a single nontrivial arm (i.e, where the other arm \
               is `_ => {}`) is used; recommends `if let` instead");
declare_lint!(pub MATCH_REF_PATS, Warn,
              "a match has all arms prefixed with `&`; the match expression can be \
               dereferenced instead");

#[allow(missing_copy_implementations)]
pub struct MatchPass;

impl LintPass for MatchPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SINGLE_MATCH, MATCH_REF_PATS)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprMatch(ref ex, ref arms, ast::MatchSource::Normal) = expr.node {
            // check preconditions for SINGLE_MATCH
                // only two arms
            if arms.len() == 2 &&
                // both of the arms have a single pattern and no guard
                arms[0].pats.len() == 1 && arms[0].guard.is_none() &&
                arms[1].pats.len() == 1 && arms[1].guard.is_none() &&
                // and the second pattern is a `_` wildcard: this is not strictly necessary,
                // since the exhaustiveness check will ensure the last one is a catch-all,
                // but in some cases, an explicit match is preferred to catch situations
                // when an enum is extended, so we don't consider these cases
                arms[1].pats[0].node == PatWild(PatWildSingle) &&
                // finally, we don't want any content in the second arm (unit or empty block)
                is_unit_expr(&arms[1].body)
            {
                if in_external_macro(cx, expr.span) {return;}
                span_help_and_lint(cx, SINGLE_MATCH, expr.span,
                      "you seem to be trying to use match for \
                      destructuring a single pattern. Did you mean to \
                      use `if let`?",
                      &format!("try\nif let {} = {} {}",
                               snippet(cx, arms[0].pats[0].span, ".."),
                               snippet(cx, ex.span, ".."),
                               expr_block(cx, &arms[0].body, "..")));
            }

            // check preconditions for MATCH_REF_PATS
            if has_only_ref_pats(arms) {
                if let ExprAddrOf(Mutability::MutImmutable, ref inner) = ex.node {
                    span_lint(cx, MATCH_REF_PATS, expr.span, &format!(
                        "you don't need to add `&` to both the expression to match \
                         and the patterns: use `match {} {{ ...`", snippet(cx, inner.span, "..")));
                } else {
                    span_lint(cx, MATCH_REF_PATS, expr.span, &format!(
                        "instead of prefixing all patterns with `&`, you can dereference the \
                         expression to match: `match *{} {{ ...`", snippet(cx, ex.span, "..")));
                }
            }
        }
    }
}

fn is_unit_expr(expr: &Expr) -> bool {
    match expr.node {
        ExprTup(ref v) if v.is_empty() => true,
        ExprBlock(ref b) if b.stmts.is_empty() && b.expr.is_none() => true,
        _ => false,
    }
}

fn has_only_ref_pats(arms: &[Arm]) -> bool {
    arms.iter().flat_map(|a| &a.pats).all(|p| match p.node {
        PatRegion(..) => true,  // &-patterns
        PatWild(..) => true,    // an "anything" wildcard is also fine
        _ => false,
    })
}

//! Checks for needless boolean results of if-else expressions
//!
//! This lint is **warn** by default

use rustc::lint::*;
use syntax::ast::*;

use utils::{de_p, span_lint, snippet};

declare_lint! {
    pub NEEDLESS_BOOL,
    Warn,
    "if-statements with plain booleans in the then- and else-clause, e.g. \
     `if p { true } else { false }`"
}

#[derive(Copy,Clone)]
pub struct NeedlessBool;

impl LintPass for NeedlessBool {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_BOOL)
    }

    fn check_expr(&mut self, cx: &Context, e: &Expr) {
        if let ExprIf(ref pred, ref then_block, Some(ref else_expr)) = e.node {
            match (fetch_bool_block(then_block), fetch_bool_expr(else_expr)) {
                (Some(true), Some(true)) => {
                    span_lint(cx, NEEDLESS_BOOL, e.span,
                              "this if-then-else expression will always return true"); },
                (Some(false), Some(false)) => {
                    span_lint(cx, NEEDLESS_BOOL, e.span,
                              "this if-then-else expression will always return false"); },
                (Some(true), Some(false)) => {
                    let pred_snip = snippet(cx, pred.span, "..");
                    let hint = if pred_snip == ".." { "its predicate".into() } else {
                        format!("`{}`", pred_snip)
                    };
                    span_lint(cx, NEEDLESS_BOOL, e.span, &format!(
                        "you can reduce this if-then-else expression to just {}", hint));
                },
                (Some(false), Some(true)) => {
                    let pred_snip = snippet(cx, pred.span, "..");
                    let hint = if pred_snip == ".." { "`!` and its predicate".into() } else {
                        format!("`!{}`", pred_snip)
                    };
                    span_lint(cx, NEEDLESS_BOOL, e.span, &format!(
                        "you can reduce this if-then-else expression to just {}", hint));
                },
                _ => ()
            }
        }
    }
}

fn fetch_bool_block(block: &Block) -> Option<bool> {
    if block.stmts.is_empty() {
        block.expr.as_ref().map(de_p).and_then(fetch_bool_expr)
    } else { None }
}

fn fetch_bool_expr(expr: &Expr) -> Option<bool> {
    match &expr.node {
        &ExprBlock(ref block) => fetch_bool_block(block),
        &ExprLit(ref lit_ptr) => if let &LitBool(value) = &lit_ptr.node {
            Some(value) } else { None },
        _ => None
    }
}

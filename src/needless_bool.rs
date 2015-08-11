//! Checks for needless boolean results of if-else expressions
//!
//! This lint is **warn** by default

use rustc::plugin::Registry;
use rustc::lint::*;
use rustc::middle::const_eval::lookup_const_by_id;
use rustc::middle::def::*;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use syntax::ptr::P;
use syntax::codemap::Span;
use utils::{de_p, span_lint};

declare_lint! {
    pub NEEDLESS_BOOL,
    Warn,
    "Warn on needless use of if x { true } else { false } (or vice versa)"
}

#[derive(Copy,Clone)]
pub struct NeedlessBool;

impl LintPass for NeedlessBool {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_BOOL)
    }

    fn check_expr(&mut self, cx: &Context, e: &Expr) {
        if let ExprIf(_, ref then_block, Option::Some(ref else_expr)) = e.node {
            match (fetch_bool_block(then_block), fetch_bool_expr(else_expr)) {
                (Option::Some(true), Option::Some(true)) => {
                    span_lint(cx, NEEDLESS_BOOL, e.span,
                              "your if-then-else expression will always return true"); },
                (Option::Some(true), Option::Some(false)) => {
                    span_lint(cx, NEEDLESS_BOOL, e.span,
                              "you can reduce your if-statement to its predicate"); },
                (Option::Some(false), Option::Some(true)) => {
                    span_lint(cx, NEEDLESS_BOOL, e.span,
                              "you can reduce your if-statement to '!' + your predicate"); },
                (Option::Some(false), Option::Some(false)) => {
                    span_lint(cx, NEEDLESS_BOOL, e.span,
                              "your if-then-else expression will always return false"); },
                _ => ()
            }
        }
    }
}

fn fetch_bool_block(block: &Block) -> Option<bool> {
    if block.stmts.is_empty() {
        block.expr.as_ref().map(de_p).and_then(fetch_bool_expr)
    } else { Option::None }
}

fn fetch_bool_expr(expr: &Expr) -> Option<bool> {
    match &expr.node {
        &ExprBlock(ref block) => fetch_bool_block(block),
        &ExprLit(ref lit_ptr) => if let &LitBool(value) = &lit_ptr.node {
            Option::Some(value) } else { Option::None },
        _ => Option::None
    }
}

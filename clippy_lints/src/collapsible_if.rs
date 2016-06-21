//! Checks for if expressions that contain only an if expression.
//!
//! For example, the lint would catch:
//!
//! ```
//! if x {
//!     if y {
//!         println!("Hello world");
//!     }
//! }
//! ```
//!
//! This lint is **warn** by default

use rustc::lint::*;
use std::borrow::Cow;
use syntax::codemap::Spanned;
use syntax::ast;

use utils::{in_macro, snippet, snippet_block, span_lint_and_then};

/// **What it does:** This lint checks for nested `if`-statements which can be collapsed by
/// `&&`-combining their conditions and for `else { if .. }` expressions that can be collapsed to
/// `else if ..`.
///
/// **Why is this bad?** Each `if`-statement adds one level of nesting, which makes code look more complex than it really is.
///
/// **Known problems:** None
///
/// **Example:** `if x { if y { .. } }`
declare_lint! {
    pub COLLAPSIBLE_IF,
    Warn,
    "two nested `if`-expressions can be collapsed into one, e.g. `if x { if y { foo() } }` \
     can be written as `if x && y { foo() }` and an `else { if .. } expression can be collapsed to \
     `else if`"
}

#[derive(Copy,Clone)]
pub struct CollapsibleIf;

impl LintPass for CollapsibleIf {
    fn get_lints(&self) -> LintArray {
        lint_array!(COLLAPSIBLE_IF)
    }
}

impl EarlyLintPass for CollapsibleIf {
    fn check_expr(&mut self, cx: &EarlyContext, expr: &ast::Expr) {
        if !in_macro(cx, expr.span) {
            check_if(cx, expr)
        }
    }
}

fn check_if(cx: &EarlyContext, expr: &ast::Expr) {
    match expr.node {
        ast::ExprKind::If(ref check, ref then, ref else_) => {
            if let Some(ref else_) = *else_ {
                check_collapsible_maybe_if_let(cx, else_);
            } else {
                check_collapsible_no_if_let(cx, expr, check, then);
            }
        }
        ast::ExprKind::IfLet(_, _, _, Some(ref else_)) => {
            check_collapsible_maybe_if_let(cx, else_);
        }
        _ => (),
    }
}

fn check_collapsible_maybe_if_let(cx: &EarlyContext, else_: &ast::Expr) {
    if_let_chain! {[
        let ast::ExprKind::Block(ref block) = else_.node,
        block.stmts.is_empty(),
        let Some(ref else_) = block.expr,
    ], {
        match else_.node {
            ast::ExprKind::If(..) | ast::ExprKind::IfLet(..) => {
                span_lint_and_then(cx,
                                   COLLAPSIBLE_IF,
                                   block.span,
                                   "this `else { if .. }` block can be collapsed", |db| {
                    db.span_suggestion(block.span, "try", snippet_block(cx, else_.span, "..").into_owned());
                });
            }
            _ => (),
        }
    }}
}

fn check_collapsible_no_if_let(
    cx: &EarlyContext,
    expr: &ast::Expr,
    check: &ast::Expr,
    then: &ast::Block,
) {
    if_let_chain! {[
        let Some(inner) = single_stmt_of_block(then),
        let ast::ExprKind::If(ref check_inner, ref content, None) = inner.node,
    ], {
        if expr.span.expn_id != inner.span.expn_id {
            return;
        }
        span_lint_and_then(cx, COLLAPSIBLE_IF, expr.span, "this if statement can be collapsed", |db| {
            db.span_suggestion(expr.span,
                               "try",
                               format!("if {} && {} {}",
                                       check_to_string(cx, check),
                                       check_to_string(cx, check_inner),
                                       snippet_block(cx, content.span, "..")));
        });
    }}
}

fn requires_brackets(e: &ast::Expr) -> bool {
    match e.node {
        ast::ExprKind::Binary(Spanned { node: n, .. }, _, _) if n == ast::BinOpKind::Eq => false,
        _ => true,
    }
}

fn check_to_string(cx: &EarlyContext, e: &ast::Expr) -> Cow<'static, str> {
    if requires_brackets(e) {
        format!("({})", snippet(cx, e.span, "..")).into()
    } else {
        snippet(cx, e.span, "..")
    }
}

fn single_stmt_of_block(block: &ast::Block) -> Option<&ast::Expr> {
    if block.stmts.len() == 1 && block.expr.is_none() {
        if let ast::StmtKind::Expr(ref expr, _) = block.stmts[0].node {
            single_stmt_of_expr(expr)
        } else {
            None
        }
    } else if block.stmts.is_empty() {
        if let Some(ref p) = block.expr {
            Some(p)
        } else {
            None
        }
    } else {
        None
    }
}

fn single_stmt_of_expr(expr: &ast::Expr) -> Option<&ast::Expr> {
    if let ast::ExprKind::Block(ref block) = expr.node {
        single_stmt_of_block(block)
    } else {
        Some(expr)
    }
}

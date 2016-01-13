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
use rustc_front::hir::*;
use syntax::codemap::Spanned;

use utils::{in_macro, snippet, snippet_block, span_lint_and_then};

/// **What it does:** This lint checks for nested `if`-statements which can be collapsed by
/// `&&`-combining their conditions and for `else { if .. } expressions that can be collapsed to
/// `else if ..`. It is `Warn` by default.
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

impl LateLintPass for CollapsibleIf {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if !in_macro(cx, expr.span) {
            check_if(cx, expr)
        }
    }
}

fn check_if(cx: &LateContext, e: &Expr) {
    if let ExprIf(ref check, ref then, ref else_) = e.node {
        match *else_ {
            Some(ref else_) => {
                if_let_chain! {[
                    let ExprBlock(ref block) = else_.node,
                    block.stmts.is_empty(),
                    block.rules == BlockCheckMode::DefaultBlock,
                    let Some(ref else_) = block.expr,
                    let ExprIf(_, _, _) = else_.node
                ], {
                    span_lint_and_then(cx,
                                       COLLAPSIBLE_IF,
                                       block.span,
                                       "this `else { if .. }` block can be collapsed", |db| {
                        db.span_suggestion(block.span, "try",
                                           format!("else {}",
                                                   snippet_block(cx, else_.span, "..")));
                    });
                }}
            }
            None => {
                if let Some(&Expr{ node: ExprIf(ref check_inner, ref content, None), span: sp, ..}) =
                       single_stmt_of_block(then) {
                    if e.span.expn_id != sp.expn_id {
                        return;
                    }
                    span_lint_and_then(cx,
                                       COLLAPSIBLE_IF,
                                       e.span,
                                       "this if statement can be collapsed", |db| {
                        db.span_suggestion(e.span, "try",
                                           format!("if {} && {} {}",
                                                   check_to_string(cx, check),
                                                   check_to_string(cx, check_inner),
                                                   snippet_block(cx, content.span, "..")));
                    });
                }
            }
        }
    }
}

fn requires_brackets(e: &Expr) -> bool {
    match e.node {
        ExprBinary(Spanned {node: n, ..}, _, _) if n == BiEq => false,
        _ => true,
    }
}

fn check_to_string(cx: &LateContext, e: &Expr) -> String {
    if requires_brackets(e) {
        format!("({})", snippet(cx, e.span, ".."))
    } else {
        format!("{}", snippet(cx, e.span, ".."))
    }
}

fn single_stmt_of_block(block: &Block) -> Option<&Expr> {
    if block.stmts.len() == 1 && block.expr.is_none() {
        if let StmtExpr(ref expr, _) = block.stmts[0].node {
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

fn single_stmt_of_expr(expr: &Expr) -> Option<&Expr> {
    if let ExprBlock(ref block) = expr.node {
        single_stmt_of_block(block)
    } else {
        Some(expr)
    }
}

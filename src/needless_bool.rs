//! Checks for needless boolean results of if-else expressions
//!
//! This lint is **warn** by default

use rustc::lint::*;
use rustc_front::hir::*;
use syntax::ast::LitKind;
use syntax::codemap::Spanned;
use utils::{span_lint, span_lint_and_then, snippet};

/// **What it does:** This lint checks for expressions of the form `if c { true } else { false }` (or vice versa) and suggest using the condition directly.
///
/// **Why is this bad?** Redundant code.
///
/// **Known problems:** Maybe false positives: Sometimes, the two branches are painstakingly documented (which we of course do not detect), so they *may* have some value. Even then, the documentation can be rewritten to match the shorter code.
///
/// **Example:** `if x { false } else { true }`
declare_lint! {
    pub NEEDLESS_BOOL,
    Warn,
    "if-statements with plain booleans in the then- and else-clause, e.g. \
     `if p { true } else { false }`"
}

/// **What it does:** This lint checks for expressions of the form `x == true` (or vice versa) and suggest using the variable directly.
///
/// **Why is this bad?** Unnecessary code.
///
/// **Known problems:** None.
///
/// **Example:** `if x == true { }` could be `if x { }`
declare_lint! {
    pub BOOL_COMPARISON,
    Warn,
    "comparing a variable to a boolean, e.g. \
     `if x == true`"
}

#[derive(Copy,Clone)]
pub struct NeedlessBool;

impl LintPass for NeedlessBool {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_BOOL)
    }
}

impl LateLintPass for NeedlessBool {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprIf(ref pred, ref then_block, Some(ref else_expr)) = e.node {
            match (fetch_bool_block(then_block), fetch_bool_expr(else_expr)) {
                (Some(true), Some(true)) => {
                    span_lint(cx,
                              NEEDLESS_BOOL,
                              e.span,
                              "this if-then-else expression will always return true");
                }
                (Some(false), Some(false)) => {
                    span_lint(cx,
                              NEEDLESS_BOOL,
                              e.span,
                              "this if-then-else expression will always return false");
                }
                (Some(true), Some(false)) => {
                    let pred_snip = snippet(cx, pred.span, "..");
                    let hint = if pred_snip == ".." {
                        "its predicate".into()
                    } else {
                        format!("`{}`", pred_snip)
                    };
                    span_lint(cx,
                              NEEDLESS_BOOL,
                              e.span,
                              &format!("you can reduce this if-then-else expression to just {}", hint));
                }
                (Some(false), Some(true)) => {
                    let pred_snip = snippet(cx, pred.span, "..");
                    let hint = if pred_snip == ".." {
                        "`!` and its predicate".into()
                    } else {
                        format!("`!{}`", pred_snip)
                    };
                    span_lint(cx,
                              NEEDLESS_BOOL,
                              e.span,
                              &format!("you can reduce this if-then-else expression to just {}", hint));
                }
                _ => (),
            }
        }
    }
}

#[derive(Copy,Clone)]
pub struct BoolComparison;

impl LintPass for BoolComparison {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOOL_COMPARISON)
    }
}

impl LateLintPass for BoolComparison {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprBinary(Spanned{ node: BiEq, .. }, ref left_side, ref right_side) = e.node {
            match (fetch_bool_expr(left_side), fetch_bool_expr(right_side)) {
                (Some(true), None) => {
                    let hint = snippet(cx, right_side.span, "..").into_owned();
                    span_lint_and_then(cx,
                                       BOOL_COMPARISON,
                                       e.span,
                                       "equality checks against true are unnecesary",
                                       |db| {
                                           db.span_suggestion(e.span, "try simplifying it as shown:", hint);
                                       });
                }
                (None, Some(true)) => {
                    let hint = snippet(cx, left_side.span, "..").into_owned();
                    span_lint_and_then(cx,
                                       BOOL_COMPARISON,
                                       e.span,
                                       "equality checks against true are unnecesary",
                                       |db| {
                                           db.span_suggestion(e.span, "try simplifying it as shown:", hint);
                                       });
                }
                (Some(false), None) => {
                    let hint = format!("!{}", snippet(cx, right_side.span, ".."));
                    span_lint_and_then(cx,
                                       BOOL_COMPARISON,
                                       e.span,
                                       "equality checks against false can be replaced by a negation",
                                       |db| {
                                           db.span_suggestion(e.span, "try simplifying it as shown:", hint);
                                       });
                }
                (None, Some(false)) => {
                    let hint = format!("!{}", snippet(cx, left_side.span, ".."));
                    span_lint_and_then(cx,
                                       BOOL_COMPARISON,
                                       e.span,
                                       "equality checks against false can be replaced by a negation",
                                       |db| {
                                           db.span_suggestion(e.span, "try simplifying it as shown:", hint);
                                       });
                }
                _ => (),
            }
        }
    }
}

fn fetch_bool_block(block: &Block) -> Option<bool> {
    if block.stmts.is_empty() {
        block.expr.as_ref().and_then(|e| fetch_bool_expr(e))
    } else {
        None
    }
}

fn fetch_bool_expr(expr: &Expr) -> Option<bool> {
    match expr.node {
        ExprBlock(ref block) => fetch_bool_block(block),
        ExprLit(ref lit_ptr) => {
            if let LitKind::Bool(value) = lit_ptr.node {
                Some(value)
            } else {
                None
            }
        }
        _ => None,
    }
}

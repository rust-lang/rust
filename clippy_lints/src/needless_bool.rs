//! Checks for needless boolean results of if-else expressions
//!
//! This lint is **warn** by default

use rustc::lint::*;
use rustc::hir::*;
use syntax::ast::LitKind;
use syntax::codemap::Spanned;
use utils::{span_lint, span_lint_and_sugg, snippet};
use utils::sugg::Sugg;

/// **What it does:** Checks for expressions of the form `if c { true } else { false }`
/// (or vice versa) and suggest using the condition directly.
///
/// **Why is this bad?** Redundant code.
///
/// **Known problems:** Maybe false positives: Sometimes, the two branches are
/// painstakingly documented (which we of course do not detect), so they *may*
/// have some value. Even then, the documentation can be rewritten to match the
/// shorter code.
///
/// **Example:**
/// ```rust
/// if x { false } else { true }
/// ```
declare_lint! {
    pub NEEDLESS_BOOL,
    Warn,
    "if-statements with plain booleans in the then- and else-clause, e.g. \
     `if p { true } else { false }`"
}

/// **What it does:** Checks for expressions of the form `x == true` (or vice
/// versa) and suggest using the variable directly.
///
/// **Why is this bad?** Unnecessary code.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// if x == true { }  // could be `if x { }`
/// ```
declare_lint! {
    pub BOOL_COMPARISON,
    Warn,
    "comparing a variable to a boolean, e.g. `if x == true`"
}

#[derive(Copy,Clone)]
pub struct NeedlessBool;

impl LintPass for NeedlessBool {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_BOOL)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NeedlessBool {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        use self::Expression::*;
        if let ExprIf(ref pred, ref then_block, Some(ref else_expr)) = e.node {
            let reduce = |ret, not| {
                let snip = Sugg::hir(cx, pred, "<predicate>");
                let snip = if not { !snip } else { snip };

                let hint = if ret {
                    format!("return {}", snip)
                } else {
                    snip.to_string()
                };

                span_lint_and_sugg(cx,
                                   NEEDLESS_BOOL,
                                   e.span,
                                   "this if-then-else expression returns a bool literal",
                                   "you can reduce it to",
                                   hint);
            };
            if let ExprBlock(ref then_block) = then_block.node {
                match (fetch_bool_block(then_block), fetch_bool_expr(else_expr)) {
                    (RetBool(true), RetBool(true)) |
                    (Bool(true), Bool(true)) => {
                        span_lint(cx,
                                  NEEDLESS_BOOL,
                                  e.span,
                                  "this if-then-else expression will always return true");
                    },
                    (RetBool(false), RetBool(false)) |
                    (Bool(false), Bool(false)) => {
                        span_lint(cx,
                                  NEEDLESS_BOOL,
                                  e.span,
                                  "this if-then-else expression will always return false");
                    },
                    (RetBool(true), RetBool(false)) => reduce(true, false),
                    (Bool(true), Bool(false)) => reduce(false, false),
                    (RetBool(false), RetBool(true)) => reduce(true, true),
                    (Bool(false), Bool(true)) => reduce(false, true),
                    _ => (),
                }
            } else {
                panic!("IfExpr 'then' node is not an ExprBlock");
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for BoolComparison {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        use self::Expression::*;
        if let ExprBinary(Spanned { node: BiEq, .. }, ref left_side, ref right_side) = e.node {
            match (fetch_bool_expr(left_side), fetch_bool_expr(right_side)) {
                (Bool(true), Other) => {
                    let hint = snippet(cx, right_side.span, "..").into_owned();
                    span_lint_and_sugg(cx,
                                       BOOL_COMPARISON,
                                       e.span,
                                       "equality checks against true are unnecessary",
                                       "try simplifying it as shown",
                                       hint);
                },
                (Other, Bool(true)) => {
                    let hint = snippet(cx, left_side.span, "..").into_owned();
                    span_lint_and_sugg(cx,
                                       BOOL_COMPARISON,
                                       e.span,
                                       "equality checks against true are unnecessary",
                                       "try simplifying it as shown",
                                       hint);
                },
                (Bool(false), Other) => {
                    let hint = Sugg::hir(cx, right_side, "..");
                    span_lint_and_sugg(cx,
                                       BOOL_COMPARISON,
                                       e.span,
                                       "equality checks against false can be replaced by a negation",
                                       "try simplifying it as shown",
                                       (!hint).to_string());
                },
                (Other, Bool(false)) => {
                    let hint = Sugg::hir(cx, left_side, "..");
                    span_lint_and_sugg(cx,
                                       BOOL_COMPARISON,
                                       e.span,
                                       "equality checks against false can be replaced by a negation",
                                       "try simplifying it as shown",
                                       (!hint).to_string());
                },
                _ => (),
            }
        }
    }
}

enum Expression {
    Bool(bool),
    RetBool(bool),
    Other,
}

fn fetch_bool_block(block: &Block) -> Expression {
    match (&*block.stmts, block.expr.as_ref()) {
        (&[], Some(e)) => fetch_bool_expr(&**e),
        (&[ref e], None) => {
            if let StmtSemi(ref e, _) = e.node {
                if let ExprRet(_) = e.node {
                    fetch_bool_expr(&**e)
                } else {
                    Expression::Other
                }
            } else {
                Expression::Other
            }
        },
        _ => Expression::Other,
    }
}

fn fetch_bool_expr(expr: &Expr) -> Expression {
    match expr.node {
        ExprBlock(ref block) => fetch_bool_block(block),
        ExprLit(ref lit_ptr) => {
            if let LitKind::Bool(value) = lit_ptr.node {
                Expression::Bool(value)
            } else {
                Expression::Other
            }
        },
        ExprRet(Some(ref expr)) => {
            match fetch_bool_expr(expr) {
                Expression::Bool(value) => Expression::RetBool(value),
                _ => Expression::Other,
            }
        },
        _ => Expression::Other,
    }
}

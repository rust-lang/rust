// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! Checks for needless boolean results of if-else expressions
//!
//! This lint is **warn** by default

use crate::rustc::hir::*;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_errors::Applicability;
use crate::syntax::ast::LitKind;
use crate::syntax::source_map::Spanned;
use crate::utils::sugg::Sugg;
use crate::utils::{in_macro, snippet_with_applicability, span_lint, span_lint_and_sugg};

/// **What it does:** Checks for expressions of the form `if c { true } else {
/// false }`
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
declare_clippy_lint! {
    pub NEEDLESS_BOOL,
    complexity,
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
declare_clippy_lint! {
    pub BOOL_COMPARISON,
    complexity,
    "comparing a variable to a boolean, e.g. `if x == true`"
}

#[derive(Copy, Clone)]
pub struct NeedlessBool;

impl LintPass for NeedlessBool {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_BOOL)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NeedlessBool {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        use self::Expression::*;
        if let ExprKind::If(ref pred, ref then_block, Some(ref else_expr)) = e.node {
            let reduce = |ret, not| {
                let mut applicability = Applicability::MachineApplicable;
                let snip = Sugg::hir_with_applicability(cx, pred, "<predicate>", &mut applicability);
                let snip = if not { !snip } else { snip };

                let hint = if ret {
                    format!("return {}", snip)
                } else {
                    snip.to_string()
                };

                span_lint_and_sugg(
                    cx,
                    NEEDLESS_BOOL,
                    e.span,
                    "this if-then-else expression returns a bool literal",
                    "you can reduce it to",
                    hint,
                    applicability,
                );
            };
            if let ExprKind::Block(ref then_block, _) = then_block.node {
                match (fetch_bool_block(then_block), fetch_bool_expr(else_expr)) {
                    (RetBool(true), RetBool(true)) | (Bool(true), Bool(true)) => {
                        span_lint(
                            cx,
                            NEEDLESS_BOOL,
                            e.span,
                            "this if-then-else expression will always return true",
                        );
                    },
                    (RetBool(false), RetBool(false)) | (Bool(false), Bool(false)) => {
                        span_lint(
                            cx,
                            NEEDLESS_BOOL,
                            e.span,
                            "this if-then-else expression will always return false",
                        );
                    },
                    (RetBool(true), RetBool(false)) => reduce(true, false),
                    (Bool(true), Bool(false)) => reduce(false, false),
                    (RetBool(false), RetBool(true)) => reduce(true, true),
                    (Bool(false), Bool(true)) => reduce(false, true),
                    _ => (),
                }
            } else {
                panic!("IfExpr 'then' node is not an ExprKind::Block");
            }
        }
    }
}

#[derive(Copy, Clone)]
pub struct BoolComparison;

impl LintPass for BoolComparison {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOOL_COMPARISON)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for BoolComparison {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        use self::Expression::*;

        if in_macro(e.span) {
            return;
        }

        if let ExprKind::Binary(Spanned { node: BinOpKind::Eq, .. }, ref left_side, ref right_side) = e.node {
            let mut applicability = Applicability::MachineApplicable;
            match (fetch_bool_expr(left_side), fetch_bool_expr(right_side)) {
                (Bool(true), Other) => {
                    let hint = snippet_with_applicability(cx, right_side.span, "..", &mut applicability);
                    span_lint_and_sugg(
                        cx,
                        BOOL_COMPARISON,
                        e.span,
                        "equality checks against true are unnecessary",
                        "try simplifying it as shown",
                        hint.to_string(),
                        applicability,
                    );
                },
                (Other, Bool(true)) => {
                    let hint = snippet_with_applicability(cx, left_side.span, "..", &mut applicability);
                    span_lint_and_sugg(
                        cx,
                        BOOL_COMPARISON,
                        e.span,
                        "equality checks against true are unnecessary",
                        "try simplifying it as shown",
                        hint.to_string(),
                        applicability,
                    );
                },
                (Bool(false), Other) => {
                    let hint = Sugg::hir_with_applicability(cx, right_side, "..", &mut applicability);
                    span_lint_and_sugg(
                        cx,
                        BOOL_COMPARISON,
                        e.span,
                        "equality checks against false can be replaced by a negation",
                        "try simplifying it as shown",
                        (!hint).to_string(),
                        applicability,
                    );
                },
                (Other, Bool(false)) => {
                    let hint = Sugg::hir_with_applicability(cx, left_side, "..", &mut applicability);
                    span_lint_and_sugg(
                        cx,
                        BOOL_COMPARISON,
                        e.span,
                        "equality checks against false can be replaced by a negation",
                        "try simplifying it as shown",
                        (!hint).to_string(),
                        applicability,
                    );
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
        (&[ref e], None) => if let StmtKind::Semi(ref e, _) = e.node {
            if let ExprKind::Ret(_) = e.node {
                fetch_bool_expr(&**e)
            } else {
                Expression::Other
            }
        } else {
            Expression::Other
        },
        _ => Expression::Other,
    }
}

fn fetch_bool_expr(expr: &Expr) -> Expression {
    match expr.node {
        ExprKind::Block(ref block, _) => fetch_bool_block(block),
        ExprKind::Lit(ref lit_ptr) => if let LitKind::Bool(value) = lit_ptr.node {
            Expression::Bool(value)
        } else {
            Expression::Other
        },
        ExprKind::Ret(Some(ref expr)) => match fetch_bool_expr(expr) {
            Expression::Bool(value) => Expression::RetBool(value),
            _ => Expression::Other,
        },
        _ => Expression::Other,
    }
}

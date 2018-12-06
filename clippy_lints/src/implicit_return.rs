// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir::{intravisit::FnKind, Body, ExprKind, FnDecl};
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_errors::Applicability;
use crate::syntax::{ast::NodeId, source_map::Span};
use crate::utils::{snippet_opt, span_lint_and_then};

/// **What it does:** Checks for missing return statements at the end of a block.
///
/// **Why is this bad?** Actually omitting the return keyword is idiomatic Rust code. Programmers
/// coming from other languages might prefer the expressiveness of `return`. It's possible to miss
/// the last returning statement because the only difference is a missing `;`. Especially in bigger
/// code with multiple return paths having a `return` keyword makes it easier to find the
/// corresponding statements.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn foo(x: usize) {
///     x
/// }
/// ```
/// add return
/// ```rust
/// fn foo(x: usize) {
///     return x;
/// }
/// ```
declare_clippy_lint! {
    pub IMPLICIT_RETURN,
    restriction,
    "use a return statement like `return expr` instead of an expression"
}

pub struct Pass;

impl Pass {
    fn expr_match(cx: &LateContext<'_, '_>, expr: &rustc::hir::Expr) {
        match &expr.node {
            ExprKind::Block(block, ..) => {
                if let Some(expr) = &block.expr {
                    Self::expr_match(cx, expr);
                }
                // only needed in the case of `break` with `;` at the end
                else if let Some(stmt) = block.stmts.last() {
                    if let rustc::hir::StmtKind::Semi(expr, ..) = &stmt.node {
                        Self::expr_match(cx, expr);
                    }
                }
            },
            // use `return` instead of `break`
            ExprKind::Break(.., break_expr) => {
                if let Some(break_expr) = break_expr {
                    span_lint_and_then(cx, IMPLICIT_RETURN, expr.span, "missing return statement", |db| {
                        if let Some(snippet) = snippet_opt(cx, break_expr.span) {
                            db.span_suggestion_with_applicability(
                                expr.span,
                                "change `break` to `return` as shown",
                                format!("return {}", snippet),
                                Applicability::MachineApplicable,
                            );
                        }
                    });
                }
            },
            ExprKind::If(.., if_expr, else_expr) => {
                Self::expr_match(cx, if_expr);

                if let Some(else_expr) = else_expr {
                    Self::expr_match(cx, else_expr);
                }
            },
            ExprKind::Match(_, arms, ..) => {
                for arm in arms {
                    Self::expr_match(cx, &arm.body);
                }
            },
            // loops could be using `break` instead of `return`
            ExprKind::Loop(block, ..) => {
                if let Some(expr) = &block.expr {
                    Self::expr_match(cx, expr);
                }
                // only needed in the case of `break` with `;` at the end
                else if let Some(stmt) = block.stmts.last() {
                    if let rustc::hir::StmtKind::Semi(expr, ..) = &stmt.node {
                        Self::expr_match(cx, expr);
                    }
                }
            },
            // skip if it already has a return statement
            ExprKind::Ret(..) => (),
            // everything else is missing `return`
            _ => span_lint_and_then(cx, IMPLICIT_RETURN, expr.span, "missing return statement", |db| {
                if let Some(snippet) = snippet_opt(cx, expr.span) {
                    db.span_suggestion_with_applicability(
                        expr.span,
                        "add `return` as shown",
                        format!("return {}", snippet),
                        Applicability::MachineApplicable,
                    );
                }
            }),
        }
    }
}

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(IMPLICIT_RETURN)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl,
        body: &'tcx Body,
        _: Span,
        _: NodeId,
    ) {
        let def_id = cx.tcx.hir.body_owner_def_id(body.id());
        let mir = cx.tcx.optimized_mir(def_id);

        // checking return type through MIR, HIR is not able to determine inferred closure return types
        if !mir.return_ty().is_unit() {
            Self::expr_match(cx, &body.value);
        }
    }
}

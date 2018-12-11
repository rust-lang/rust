// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir::def::Def;
use crate::rustc::hir::*;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::syntax::ptr::P;
use crate::utils::sugg::Sugg;
use if_chain::if_chain;

use crate::rustc_errors::Applicability;
use crate::utils::paths::*;
use crate::utils::{match_def_path, match_type, span_lint_and_then};

/// **What it does:** Checks for expressions that could be replaced by the question mark operator
///
/// **Why is this bad?** Question mark usage is more idiomatic
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// if option.is_none() {
///     return None;
/// }
/// ```
///
/// Could be written:
///
/// ```rust
/// option?;
/// ```
declare_clippy_lint! {
    pub QUESTION_MARK,
    style,
    "checks for expressions that could be replaced by the question mark operator"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(QUESTION_MARK)
    }
}

impl Pass {
    /// Check if the given expression on the given context matches the following structure:
    ///
    /// ```ignore
    /// if option.is_none() {
    ///    return None;
    /// }
    /// ```
    ///
    /// If it matches, it will suggest to use the question mark operator instead
    fn check_is_none_and_early_return_none(cx: &LateContext<'_, '_>, expr: &Expr) {
        if_chain! {
            if let ExprKind::If(ref if_expr, ref body, _) = expr.node;
            if let ExprKind::MethodCall(ref segment, _, ref args) = if_expr.node;
            if segment.ident.name == "is_none";
            if Self::expression_returns_none(cx, body);
            if let Some(subject) = args.get(0);
            if Self::is_option(cx, subject);

            then {
                span_lint_and_then(
                    cx,
                    QUESTION_MARK,
                    expr.span,
                    "this block may be rewritten with the `?` operator",
                    |db| {
                        let receiver_str = &Sugg::hir(cx, subject, "..");

                        db.span_suggestion_with_applicability(
                            expr.span,
                            "replace_it_with",
                            format!("{}?;", receiver_str),
                            Applicability::MaybeIncorrect, // snippet
                        );
                    }
                )
            }
        }
    }

    fn is_option(cx: &LateContext<'_, '_>, expression: &Expr) -> bool {
        let expr_ty = cx.tables.expr_ty(expression);

        match_type(cx, expr_ty, &OPTION)
    }

    fn expression_returns_none(cx: &LateContext<'_, '_>, expression: &Expr) -> bool {
        match expression.node {
            ExprKind::Block(ref block, _) => {
                if let Some(return_expression) = Self::return_expression(block) {
                    return Self::expression_returns_none(cx, &return_expression);
                }

                false
            },
            ExprKind::Ret(Some(ref expr)) => Self::expression_returns_none(cx, expr),
            ExprKind::Path(ref qp) => {
                if let Def::VariantCtor(def_id, _) = cx.tables.qpath_def(qp, expression.hir_id) {
                    return match_def_path(cx.tcx, def_id, &OPTION_NONE);
                }

                false
            },
            _ => false,
        }
    }

    fn return_expression(block: &Block) -> Option<P<Expr>> {
        // Check if last expression is a return statement. Then, return the expression
        if_chain! {
            if block.stmts.len() == 1;
            if let Some(expr) = block.stmts.iter().last();
            if let StmtKind::Semi(ref expr, _) = expr.node;
            if let ExprKind::Ret(ref ret_expr) = expr.node;
            if let &Some(ref ret_expr) = ret_expr;

            then {
                return Some(ret_expr.clone());
            }
        }

        // Check if the block has an implicit return expression
        if let Some(ref ret_expr) = block.expr {
            return Some(ret_expr.clone());
        }

        None
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        Self::check_is_none_and_early_return_none(cx, expr);
    }
}

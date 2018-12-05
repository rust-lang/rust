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
/// coming from other languages might prefer the expressiveness of `return`.
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
    pub MISSING_RETURNS,
    restriction,
    "use a return statement like `return expr` instead of an expression"
}

pub struct Pass;

impl Pass {
    fn show_suggestion(cx: &LateContext<'_, '_>, span: syntax_pos::Span) {
        span_lint_and_then(cx, MISSING_RETURNS, span, "missing return statement", |db| {
            if let Some(snippet) = snippet_opt(cx, span) {
                db.span_suggestion_with_applicability(
                    span,
                    "add `return` as shown",
                    format!("return {}", snippet),
                    Applicability::MachineApplicable,
                );
            }
        });
    }

    fn expr_match(cx: &LateContext<'_, '_>, kind: &ExprKind) {
        match kind {
            ExprKind::Block(ref block, ..) => {
                if let Some(ref expr) = block.expr {
                    Self::expr_match(cx, &expr.node);
                }
            },
            ExprKind::If(.., if_expr, else_expr) => {
                Self::expr_match(cx, &if_expr.node);

                if let Some(else_expr) = else_expr {
                    Self::expr_match(cx, &else_expr.node);
                }
            },
            ExprKind::Match(_, arms, ..) => {
                for arm in arms {
                    Self::expr_match(cx, &arm.body.node);
                }
            },
            ExprKind::Lit(lit) => Self::show_suggestion(cx, lit.span),
            _ => (),
        }
    }
}

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(MISSING_RETURNS)
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

        if !mir.return_ty().is_unit() {
            Self::expr_match(cx, &body.value.node);
        }
    }
}

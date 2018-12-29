// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! lint on if expressions with an else if, but without a final else branch

use rustc::lint::{in_external_macro, EarlyContext, EarlyLintPass, LintArray, LintContext, LintPass};
use rustc::{declare_tool_lint, lint_array};
use syntax::ast::*;

use crate::utils::span_help_and_lint;

/// **What it does:** Checks for usage of if expressions with an `else if` branch,
/// but without a final `else` branch.
///
/// **Why is this bad?** Some coding guidelines require this (e.g. MISRA-C:2004 Rule 14.10).
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// if x.is_positive() {
///     a();
/// } else if x.is_negative() {
///     b();
/// }
/// ```
///
/// Could be written:
///
/// ```rust
/// if x.is_positive() {
///     a();
/// } else if x.is_negative() {
///     b();
/// } else {
///     // we don't care about zero
/// }
/// ```
declare_clippy_lint! {
    pub ELSE_IF_WITHOUT_ELSE,
    restriction,
    "if expression with an `else if`, but without a final `else` branch"
}

#[derive(Copy, Clone)]
pub struct ElseIfWithoutElse;

impl LintPass for ElseIfWithoutElse {
    fn get_lints(&self) -> LintArray {
        lint_array!(ELSE_IF_WITHOUT_ELSE)
    }
}

impl EarlyLintPass for ElseIfWithoutElse {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, mut item: &Expr) {
        if in_external_macro(cx.sess(), item.span) {
            return;
        }

        while let ExprKind::If(_, _, Some(ref els)) = item.node {
            if let ExprKind::If(_, _, None) = els.node {
                span_help_and_lint(
                    cx,
                    ELSE_IF_WITHOUT_ELSE,
                    els.span,
                    "if expression with an `else if`, but without a final `else`",
                    "add an `else` block here",
                );
            }

            item = els;
        }
    }
}

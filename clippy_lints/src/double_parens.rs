// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::syntax::ast::*;
use crate::utils::{in_macro, span_lint};

/// **What it does:** Checks for unnecessary double parentheses.
///
/// **Why is this bad?** This makes code harder to read and might indicate a
/// mistake.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// ((0))
/// foo((0))
/// ((1, 2))
/// ```
declare_clippy_lint! {
    pub DOUBLE_PARENS,
    complexity,
    "Warn on unnecessary double parentheses"
}

#[derive(Copy, Clone)]
pub struct DoubleParens;

impl LintPass for DoubleParens {
    fn get_lints(&self) -> LintArray {
        lint_array!(DOUBLE_PARENS)
    }
}

impl EarlyLintPass for DoubleParens {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if in_macro(expr.span) {
            return;
        }

        match expr.node {
            ExprKind::Paren(ref in_paren) => match in_paren.node {
                ExprKind::Paren(_) | ExprKind::Tup(_) => {
                    span_lint(
                        cx,
                        DOUBLE_PARENS,
                        expr.span,
                        "Consider removing unnecessary double parentheses",
                    );
                },
                _ => {},
            },
            ExprKind::Call(_, ref params) => {
                if params.len() == 1 {
                    let param = &params[0];
                    if let ExprKind::Paren(_) = param.node {
                        span_lint(
                            cx,
                            DOUBLE_PARENS,
                            param.span,
                            "Consider removing unnecessary double parentheses",
                        );
                    }
                }
            },
            ExprKind::MethodCall(_, ref params) => {
                if params.len() == 2 {
                    let param = &params[1];
                    if let ExprKind::Paren(_) = param.node {
                        span_lint(
                            cx,
                            DOUBLE_PARENS,
                            param.span,
                            "Consider removing unnecessary double parentheses",
                        );
                    }
                }
            },
            _ => {},
        }
    }
}

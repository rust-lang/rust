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
use crate::rustc_errors::Applicability;
use crate::syntax::ast::*;
use crate::utils::span_lint_and_sugg;

/// **What it does:** Checks for fields in struct literals where shorthands
/// could be used.
///
/// **Why is this bad?** If the field and variable names are the same,
/// the field name is redundant.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let bar: u8 = 123;
///
/// struct Foo {
///     bar: u8,
/// }
///
/// let foo = Foo{ bar: bar }
/// ```
/// the last line can be simplified to
/// ```rust
/// let foo = Foo{ bar }
/// ```
declare_clippy_lint! {
    pub REDUNDANT_FIELD_NAMES,
    style,
    "checks for fields in struct literals where shorthands could be used"
}

pub struct RedundantFieldNames;

impl LintPass for RedundantFieldNames {
    fn get_lints(&self) -> LintArray {
        lint_array!(REDUNDANT_FIELD_NAMES)
    }
}

impl EarlyLintPass for RedundantFieldNames {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::Struct(_, ref fields, _) = expr.node {
            for field in fields {
                if field.is_shorthand {
                    continue;
                }
                if let ExprKind::Path(None, path) = &field.expr.node {
                    if path.segments.len() == 1
                        && path.segments[0].ident == field.ident
                        && path.segments[0].args.is_none()
                    {
                        span_lint_and_sugg(
                            cx,
                            REDUNDANT_FIELD_NAMES,
                            field.span,
                            "redundant field names in struct initialization",
                            "replace it with",
                            field.ident.to_string(),
                            Applicability::MachineApplicable,
                        );
                    }
                }
            }
        }
    }
}

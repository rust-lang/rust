// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir::{Expr, ExprKind};
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::syntax::ast::LitKind;
use crate::utils::{is_direct_expn_of, span_lint};
use if_chain::if_chain;

/// **What it does:** Check explicit call assert!(true)
///
/// **Why is this bad?** Will be optimized out by the compiler
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// assert!(true)
/// ```
declare_clippy_lint! {
    pub EXPLICIT_TRUE,
    correctness,
    "assert!(true) will be optimized out by the compiler"
}

/// **What it does:** Check explicit call assert!(false)
///
/// **Why is this bad?** Should probably be replaced by a panic!() or unreachable!()
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// assert!(false)
/// ```
declare_clippy_lint! {
    pub EXPLICIT_FALSE,
    correctness,
    "assert!(false) should probably be replaced by a panic!() or unreachable!()"
}

pub struct AssertChecks;

impl LintPass for AssertChecks {
    fn get_lints(&self) -> LintArray {
        lint_array![EXPLICIT_TRUE, EXPLICIT_FALSE]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AssertChecks {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if_chain! {
            if is_direct_expn_of(e.span, "assert").is_some();
            if let ExprKind::Unary(_, ref lit) = e.node;
            if let ExprKind::Lit(ref inner) = lit.node;
            then {
                match inner.node {
                    LitKind::Bool(true) => {
                        span_lint(cx, EXPLICIT_TRUE, e.span,
                            "assert!(true) will be optimized out by the compiler");
                    },
                    LitKind::Bool(false) => {
                        span_lint(cx, EXPLICIT_FALSE, e.span,
                            "assert!(false) should probably be replaced by a panic!() or unreachable!()");
                    },
                    _ => (),
                }
            }
        }
    }
}

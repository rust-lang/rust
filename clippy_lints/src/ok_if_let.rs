// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir::*;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::utils::{match_type, method_chain_args, paths, snippet, span_help_and_lint};
use if_chain::if_chain;

/// **What it does:*** Checks for unnecessary `ok()` in if let.
///
/// **Why is this bad?** Calling `ok()` in if let is unnecessary, instead match
/// on `Ok(pat)`
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for result in iter {
///     if let Some(bench) = try!(result).parse().ok() {
///         vec.push(bench)
///     }
/// }
/// ```
/// Could be written:
///
/// ```rust
/// for result in iter {
///     if let Ok(bench) = try!(result).parse() {
///         vec.push(bench)
///     }
/// }
/// ```
declare_clippy_lint! {
    pub IF_LET_SOME_RESULT,
    style,
    "usage of `ok()` in `if let Some(pat)` statements is unnecessary, match on `Ok(pat)` instead"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(IF_LET_SOME_RESULT)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! { //begin checking variables
            if let ExprKind::Match(ref op, ref body, ref source) = expr.node; //test if expr is a match
            if let MatchSource::IfLetDesugar { .. } = *source; //test if it is an If Let
            if let ExprKind::MethodCall(_, _, ref result_types) = op.node; //check is expr.ok() has type Result<T,E>.ok()
            if let PatKind::TupleStruct(QPath::Resolved(_, ref x), ref y, _)  = body[0].pats[0].node; //get operation
            if method_chain_args(op, &["ok"]).is_some(); //test to see if using ok() methoduse std::marker::Sized;

            then {
                let is_result_type = match_type(cx, cx.tables.expr_ty(&result_types[0]), &paths::RESULT);
                let some_expr_string = snippet(cx, y[0].span, "");
                if print::to_string(print::NO_ANN, |s| s.print_path(x, false)) == "Some" && is_result_type {
                    span_help_and_lint(cx, IF_LET_SOME_RESULT, expr.span,
                    "Matching on `Some` with `ok()` is redundant",
                    &format!("Consider matching on `Ok({})` and removing the call to `ok` instead", some_expr_string));
                }
            }
        }
    }
}

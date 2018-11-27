// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use super::utils::{get_arg_name, match_var, remove_blocks, snippet_with_applicability, span_lint_and_sugg};
use crate::rustc::hir::*;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_errors::Applicability;
use if_chain::if_chain;

/// **What it does:** Checks for matches being used to destructure a single-variant enum
/// or tuple struct where a `let` will suffice.
///
/// **Why is this bad?** Just readability – `let` doesn't nest, whereas a `match` does.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// enum Wrapper {
///     Data(i32),
/// }
///
/// let wrapper = Wrapper::Data(42);
///
/// let data = match wrapper {
///     Wrapper::Data(i) => i,
/// };
/// ```
///
/// The correct use would be:
/// ```rust
/// enum Wrapper {
///     Data(i32),
/// }
///
/// let wrapper = Wrapper::Data(42);
/// let Wrapper::Data(data) = wrapper;
/// ```
declare_clippy_lint! {
    pub INFALLIBLE_DESTRUCTURING_MATCH,
    style,
    "a match statement with a single infallible arm instead of a `let`"
}

#[derive(Copy, Clone, Default)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INFALLIBLE_DESTRUCTURING_MATCH)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_local(&mut self, cx: &LateContext<'a, 'tcx>, local: &'tcx Local) {
        if_chain! {
            if let Some(ref expr) = local.init;
            if let ExprKind::Match(ref target, ref arms, MatchSource::Normal) = expr.node;
            if arms.len() == 1 && arms[0].pats.len() == 1 && arms[0].guard.is_none();
            if let PatKind::TupleStruct(QPath::Resolved(None, ref variant_name), ref args, _) = arms[0].pats[0].node;
            if args.len() == 1;
            if let Some(arg) = get_arg_name(&args[0]);
            let body = remove_blocks(&arms[0].body);
            if match_var(body, arg);

            then {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    INFALLIBLE_DESTRUCTURING_MATCH,
                    local.span,
                    "you seem to be trying to use match to destructure a single infallible pattern. \
                     Consider using `let`",
                    "try this",
                    format!(
                        "let {}({}) = {};",
                        snippet_with_applicability(cx, variant_name.span, "..", &mut applicability),
                        snippet_with_applicability(cx, local.pat.span, "..", &mut applicability),
                        snippet_with_applicability(cx, target.span, "..", &mut applicability),
                    ),
                    applicability,
                );
            }
        }
    }
}

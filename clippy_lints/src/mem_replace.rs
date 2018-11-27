// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir::{Expr, ExprKind, MutMutable, QPath};
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_errors::Applicability;
use crate::utils::{match_def_path, match_qpath, opt_def_id, paths, snippet_with_applicability, span_lint_and_sugg};
use if_chain::if_chain;

/// **What it does:** Checks for `mem::replace()` on an `Option` with
/// `None`.
///
/// **Why is this bad?** `Option` already has the method `take()` for
/// taking its current value (Some(..) or None) and replacing it with
/// `None`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let mut an_option = Some(0);
/// let replaced = mem::replace(&mut an_option, None);
/// ```
/// Is better expressed with:
/// ```rust
/// let mut an_option = Some(0);
/// let taken = an_option.take();
/// ```
declare_clippy_lint! {
    pub MEM_REPLACE_OPTION_WITH_NONE,
    style,
    "replacing an `Option` with `None` instead of `take()`"
}

pub struct MemReplace;

impl LintPass for MemReplace {
    fn get_lints(&self) -> LintArray {
        lint_array![MEM_REPLACE_OPTION_WITH_NONE]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MemReplace {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            // Check that `expr` is a call to `mem::replace()`
            if let ExprKind::Call(ref func, ref func_args) = expr.node;
            if func_args.len() == 2;
            if let ExprKind::Path(ref func_qpath) = func.node;
            if let Some(def_id) = opt_def_id(cx.tables.qpath_def(func_qpath, func.hir_id));
            if match_def_path(cx.tcx, def_id, &paths::MEM_REPLACE);

            // Check that second argument is `Option::None`
            if let ExprKind::Path(ref replacement_qpath) = func_args[1].node;
            if match_qpath(replacement_qpath, &paths::OPTION_NONE);

            then {
                // Since this is a late pass (already type-checked),
                // and we already know that the second argument is an
                // `Option`, we do not need to check the first
                // argument's type. All that's left is to get
                // replacee's path.
                let replaced_path = match func_args[0].node {
                    ExprKind::AddrOf(MutMutable, ref replaced) => {
                        if let ExprKind::Path(QPath::Resolved(None, ref replaced_path)) = replaced.node {
                            replaced_path
                        } else {
                            return
                        }
                    },
                    ExprKind::Path(QPath::Resolved(None, ref replaced_path)) => replaced_path,
                    _ => return,
                };

                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    MEM_REPLACE_OPTION_WITH_NONE,
                    expr.span,
                    "replacing an `Option` with `None`",
                    "consider `Option::take()` instead",
                    format!("{}.take()", snippet_with_applicability(cx, replaced_path.span, "", &mut applicability)),
                    applicability,
                );
            }
        }
    }
}

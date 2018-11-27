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
use crate::rustc::ty::TyKind;
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc_errors::Applicability;
use if_chain::if_chain;

use crate::utils::{any_parent_is_automatically_derived, match_def_path, opt_def_id, paths, span_lint_and_sugg};


/// **What it does:** Checks for literal calls to `Default::default()`.
///
/// **Why is this bad?** It's more clear to the reader to use the name of the type whose default is
/// being gotten than the generic `Default`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// // Bad
/// let s: String = Default::default();
///
/// // Good
/// let s = String::default();
/// ```
declare_clippy_lint! {
    pub DEFAULT_TRAIT_ACCESS,
    pedantic,
    "checks for literal calls to Default::default()"
}

#[derive(Copy, Clone)]
pub struct DefaultTraitAccess;

impl LintPass for DefaultTraitAccess {
    fn get_lints(&self) -> LintArray {
        lint_array!(DEFAULT_TRAIT_ACCESS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for DefaultTraitAccess {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            if let ExprKind::Call(ref path, ..) = expr.node;
            if !any_parent_is_automatically_derived(cx.tcx, expr.id);
            if let ExprKind::Path(ref qpath) = path.node;
            if let Some(def_id) = opt_def_id(cx.tables.qpath_def(qpath, path.hir_id));
            if match_def_path(cx.tcx, def_id, &paths::DEFAULT_TRAIT_METHOD);
            then {
                match qpath {
                    QPath::Resolved(..) => {
                        if_chain! {
                            // Detect and ignore <Foo as Default>::default() because these calls do
                            // explicitly name the type.
                            if let ExprKind::Call(ref method, ref _args) = expr.node;
                            if let ExprKind::Path(ref p) = method.node;
                            if let QPath::Resolved(Some(_ty), _path) = p;
                            then {
                                return;
                            }
                        }

                        // TODO: Work out a way to put "whatever the imported way of referencing
                        // this type in this file" rather than a fully-qualified type.
                        let expr_ty = cx.tables.expr_ty(expr);
                        if let TyKind::Adt(..) = expr_ty.sty {
                            let replacement = format!("{}::default()", expr_ty);
                            span_lint_and_sugg(
                                cx,
                                DEFAULT_TRAIT_ACCESS,
                                expr.span,
                                &format!("Calling {} is more clear than this expression", replacement),
                                "try",
                                replacement,
                                Applicability::Unspecified, // First resolve the TODO above
                            );
                         }
                    },
                    QPath::TypeRelative(..) => {},
                }
            }
         }
    }
}

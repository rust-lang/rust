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
use crate::rustc_errors::Applicability;
use crate::utils::{match_def_path, opt_def_id, paths, snippet, span_lint_and_then, walk_ptrs_ty_depth};
use if_chain::if_chain;

use std::iter;

/// **What it does:** Checks for calls of `mem::discriminant()` on a non-enum type.
///
/// **Why is this bad?** The value of `mem::discriminant()` on non-enum types
/// is unspecified.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// mem::discriminant(&"hello");
/// mem::discriminant(&&Some(2));
/// ```
declare_clippy_lint! {
    pub MEM_DISCRIMINANT_NON_ENUM,
    correctness,
    "calling mem::descriminant on non-enum type"
}

pub struct MemDiscriminant;

impl LintPass for MemDiscriminant {
    fn get_lints(&self) -> LintArray {
        lint_array![MEM_DISCRIMINANT_NON_ENUM]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MemDiscriminant {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            if let ExprKind::Call(ref func, ref func_args) = expr.node;
            // is `mem::discriminant`
            if let ExprKind::Path(ref func_qpath) = func.node;
            if let Some(def_id) = opt_def_id(cx.tables.qpath_def(func_qpath, func.hir_id));
            if match_def_path(cx.tcx, def_id, &paths::MEM_DISCRIMINANT);
            // type is non-enum
            let ty_param = cx.tables.node_substs(func.hir_id).type_at(0);
            if !ty_param.is_enum();

            then {
                span_lint_and_then(
                    cx,
                    MEM_DISCRIMINANT_NON_ENUM,
                    expr.span,
                    &format!("calling `mem::discriminant` on non-enum type `{}`", ty_param),
                    |db| {
                        // if this is a reference to an enum, suggest dereferencing
                        let (base_ty, ptr_depth) = walk_ptrs_ty_depth(ty_param);
                        if ptr_depth >= 1 && base_ty.is_enum() {
                            let param = &func_args[0];

                            // cancel out '&'s first
                            let mut derefs_needed = ptr_depth;
                            let mut cur_expr = param;
                            while derefs_needed > 0  {
                                if let ExprKind::AddrOf(_, ref inner_expr) = cur_expr.node {
                                    derefs_needed -= 1;
                                    cur_expr = inner_expr;
                                } else {
                                    break;
                                }
                            }

                            let derefs: String = iter::repeat('*').take(derefs_needed).collect();
                            db.span_suggestion_with_applicability(
                                param.span,
                                "try dereferencing",
                                format!("{}{}", derefs, snippet(cx, cur_expr.span, "<param>")),
                                Applicability::MachineApplicable,
                            );
                        }
                    },
                )
            }
        }
    }
}

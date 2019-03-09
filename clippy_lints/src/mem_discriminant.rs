use crate::utils::{match_def_path, paths, snippet, span_lint_and_then, walk_ptrs_ty_depth};
use if_chain::if_chain;
use rustc::hir::{Expr, ExprKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};
use rustc_errors::Applicability;

use std::iter;

declare_clippy_lint! {
    /// **What it does:** Checks for calls of `mem::discriminant()` on a non-enum type.
    ///
    /// **Why is this bad?** The value of `mem::discriminant()` on non-enum types
    /// is unspecified.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// use std::mem;
    ///
    /// mem::discriminant(&"hello");
    /// mem::discriminant(&&Some(2));
    /// ```
    pub MEM_DISCRIMINANT_NON_ENUM,
    correctness,
    "calling mem::descriminant on non-enum type"
}

pub struct MemDiscriminant;

impl LintPass for MemDiscriminant {
    fn get_lints(&self) -> LintArray {
        lint_array![MEM_DISCRIMINANT_NON_ENUM]
    }

    fn name(&self) -> &'static str {
        "MemDiscriminant"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MemDiscriminant {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            if let ExprKind::Call(ref func, ref func_args) = expr.node;
            // is `mem::discriminant`
            if let ExprKind::Path(ref func_qpath) = func.node;
            if let Some(def_id) = cx.tables.qpath_def(func_qpath, func.hir_id).opt_def_id();
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
                            db.span_suggestion(
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

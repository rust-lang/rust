use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::{any_parent_is_automatically_derived, match_def_path, paths, span_lint_and_sugg};

declare_clippy_lint! {
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
    pub DEFAULT_TRAIT_ACCESS,
    pedantic,
    "checks for literal calls to `Default::default()`"
}

declare_lint_pass!(DefaultTraitAccess => [DEFAULT_TRAIT_ACCESS]);

impl<'tcx> LateLintPass<'tcx> for DefaultTraitAccess {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(ref path, ..) = expr.kind;
            if !any_parent_is_automatically_derived(cx.tcx, expr.hir_id);
            if let ExprKind::Path(ref qpath) = path.kind;
            if let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::DEFAULT_TRAIT_METHOD);
            then {
                match qpath {
                    QPath::Resolved(..) => {
                        if_chain! {
                            // Detect and ignore <Foo as Default>::default() because these calls do
                            // explicitly name the type.
                            if let ExprKind::Call(ref method, ref _args) = expr.kind;
                            if let ExprKind::Path(ref p) = method.kind;
                            if let QPath::Resolved(Some(_ty), _path) = p;
                            then {
                                return;
                            }
                        }

                        // TODO: Work out a way to put "whatever the imported way of referencing
                        // this type in this file" rather than a fully-qualified type.
                        let expr_ty = cx.typeck_results().expr_ty(expr);
                        if let ty::Adt(..) = expr_ty.kind {
                            let replacement = format!("{}::default()", expr_ty);
                            span_lint_and_sugg(
                                cx,
                                DEFAULT_TRAIT_ACCESS,
                                expr.span,
                                &format!("calling `{}` is more clear than this expression", replacement),
                                "try",
                                replacement,
                                Applicability::Unspecified, // First resolve the TODO above
                            );
                         }
                    },
                    QPath::TypeRelative(..) | QPath::LangItem(..) => {},
                }
            }
        }
    }
}

use rustc::hir::*;
use rustc::lint::*;
use rustc::{declare_lint, lint_array};
use if_chain::if_chain;
use rustc::ty::TypeVariants;

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
                        // TODO: Work out a way to put "whatever the imported way of referencing
                        // this type in this file" rather than a fully-qualified type.
                        let expr_ty = cx.tables.expr_ty(expr);
                        if let TypeVariants::TyAdt(..) = expr_ty.sty {
                            let replacement = format!("{}::default()", expr_ty);
                            span_lint_and_sugg(
                                cx,
                                DEFAULT_TRAIT_ACCESS,
                                expr.span,
                                &format!("Calling {} is more clear than this expression", replacement),
                                "try",
                                replacement);
                         }
                    },
                    QPath::TypeRelative(..) => {},
                }
            }
         }
    }
}

use crate::rustc::hir::{Expr, ExprKind, MutMutable, QPath};
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::utils::{match_def_path, match_qpath, match_type, opt_def_id, paths, snippet, span_lint_and_sugg};
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
/// let an_option = Some(0);
/// let replaced = mem::replace(&mut an_option, None);
/// ```
/// Is better expressed with:
/// ```rust
/// let an_option = Some(0);
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
            if let ExprKind::Call(ref func, ref func_args) = expr.node;
            if func_args.len() == 2;
            if let ExprKind::Path(ref func_qpath) = func.node;
            if let Some(def_id) = opt_def_id(cx.tables.qpath_def(func_qpath, func.hir_id));
            if match_def_path(cx.tcx, def_id, &paths::MEM_REPLACE);
            if let ExprKind::AddrOf(MutMutable, ref replaced) = func_args[0].node;
            if match_type(cx, cx.tables.expr_ty(replaced), &paths::OPTION);
            if let ExprKind::Path(ref replacement_qpath) = func_args[1].node;
            if match_qpath(replacement_qpath, &paths::OPTION_NONE);
            if let ExprKind::Path(QPath::Resolved(None, ref replaced_path)) = replaced.node;
            then {
                let sugg = format!("{}.take()", snippet(cx, replaced_path.span, ""));
                span_lint_and_sugg(
                    cx,
                    MEM_REPLACE_OPTION_WITH_NONE,
                    expr.span,
                    "replacing an `Option` with `None`",
                    "consider `Option::take()` instead",
                    sugg
                );
            }
        }
    }
}

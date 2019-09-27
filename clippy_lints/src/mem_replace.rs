use crate::utils::{
    match_def_path, match_qpath, paths, snippet_with_applicability, span_help_and_lint, span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc::hir::{Expr, ExprKind, MutMutable, QPath};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;

declare_clippy_lint! {
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
    /// use std::mem;
    ///
    /// let mut an_option = Some(0);
    /// let replaced = mem::replace(&mut an_option, None);
    /// ```
    /// Is better expressed with:
    /// ```rust
    /// let mut an_option = Some(0);
    /// let taken = an_option.take();
    /// ```
    pub MEM_REPLACE_OPTION_WITH_NONE,
    style,
    "replacing an `Option` with `None` instead of `take()`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `mem::replace(&mut _, mem::uninitialized())`
    /// and `mem::replace(&mut _, mem::zeroed())`.
    ///
    /// **Why is this bad?** This will lead to undefined behavior even if the
    /// value is overwritten later, because the uninitialized value may be
    /// observed in the case of a panic.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```
    /// use std::mem;
    ///# fn may_panic(v: Vec<i32>) -> Vec<i32> { v }
    ///
    /// #[allow(deprecated, invalid_value)]
    /// fn myfunc (v: &mut Vec<i32>) {
    ///     let taken_v = unsafe { mem::replace(v, mem::uninitialized()) };
    ///     let new_v = may_panic(taken_v); // undefined behavior on panic
    ///     mem::forget(mem::replace(v, new_v));
    /// }
    /// ```
    ///
    /// The [take_mut](https://docs.rs/take_mut) crate offers a sound solution,
    /// at the cost of either lazily creating a replacement value or aborting
    /// on panic, to ensure that the uninitialized value cannot be observed.
    pub MEM_REPLACE_WITH_UNINIT,
    correctness,
    "`mem::replace(&mut _, mem::uninitialized())` or `mem::replace(&mut _, mem::zeroed())`"
}

declare_lint_pass!(MemReplace =>
    [MEM_REPLACE_OPTION_WITH_NONE, MEM_REPLACE_WITH_UNINIT]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MemReplace {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            // Check that `expr` is a call to `mem::replace()`
            if let ExprKind::Call(ref func, ref func_args) = expr.kind;
            if func_args.len() == 2;
            if let ExprKind::Path(ref func_qpath) = func.kind;
            if let Some(def_id) = cx.tables.qpath_res(func_qpath, func.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::MEM_REPLACE);

            // Check that second argument is `Option::None`
            then {
                if let ExprKind::Path(ref replacement_qpath) = func_args[1].kind {
                    if match_qpath(replacement_qpath, &paths::OPTION_NONE) {

                        // Since this is a late pass (already type-checked),
                        // and we already know that the second argument is an
                        // `Option`, we do not need to check the first
                        // argument's type. All that's left is to get
                        // replacee's path.
                        let replaced_path = match func_args[0].kind {
                            ExprKind::AddrOf(MutMutable, ref replaced) => {
                                if let ExprKind::Path(QPath::Resolved(None, ref replaced_path)) = replaced.kind {
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
                if let ExprKind::Call(ref repl_func, ref repl_args) = func_args[1].kind {
                    if_chain! {
                        if repl_args.is_empty();
                        if let ExprKind::Path(ref repl_func_qpath) = repl_func.kind;
                        if let Some(repl_def_id) = cx.tables.qpath_res(repl_func_qpath, repl_func.hir_id).opt_def_id();
                        then {
                            if match_def_path(cx, repl_def_id, &paths::MEM_UNINITIALIZED) {
                                span_help_and_lint(
                                    cx,
                                    MEM_REPLACE_WITH_UNINIT,
                                    expr.span,
                                    "replacing with `mem::uninitialized()`",
                                    "consider using the `take_mut` crate instead",
                                );
                            } else if match_def_path(cx, repl_def_id, &paths::MEM_ZEROED) &&
                                    !cx.tables.expr_ty(&func_args[1]).is_primitive() {
                                span_help_and_lint(
                                    cx,
                                    MEM_REPLACE_WITH_UNINIT,
                                    expr.span,
                                    "replacing with `mem::zeroed()`",
                                    "consider using a default value or the `take_mut` crate instead",
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

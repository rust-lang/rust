use if_chain::if_chain;
use rustc::lint::in_external_macro;
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::{is_must_use_func_call, is_must_use_ty, span_help_and_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `let _ = <expr>`
    /// where expr is #[must_use]
    ///
    /// **Why is this bad?** It's better to explicitly
    /// handle the value of a #[must_use] expr
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn f() -> Result<u32, u32> {
    ///     Ok(0)
    /// }
    ///
    /// let _ = f();
    /// // is_ok() is marked #[must_use]
    /// let _ = f().is_ok();
    /// ```
    pub LET_UNDERSCORE_MUST_USE,
    restriction,
    "non-binding let on a `#[must_use]` expression"
}

declare_lint_pass!(LetUnderscore => [LET_UNDERSCORE_MUST_USE]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LetUnderscore {
    fn check_stmt(&mut self, cx: &LateContext<'_, '_>, stmt: &Stmt<'_>) {
        if in_external_macro(cx.tcx.sess, stmt.span) {
            return;
        }

        if_chain! {
            if let StmtKind::Local(ref local) = stmt.kind;
            if let PatKind::Wild = local.pat.kind;
            if let Some(ref init) = local.init;
            then {
                if is_must_use_ty(cx, cx.tables.expr_ty(init)) {
                   span_help_and_lint(
                        cx,
                        LET_UNDERSCORE_MUST_USE,
                        stmt.span,
                        "non-binding let on an expression with `#[must_use]` type",
                        "consider explicitly using expression value"
                    )
                } else if is_must_use_func_call(cx, init) {
                    span_help_and_lint(
                        cx,
                        LET_UNDERSCORE_MUST_USE,
                        stmt.span,
                        "non-binding let on a result of a `#[must_use]` function",
                        "consider explicitly using function result"
                    )
                }
            }
        }
    }
}

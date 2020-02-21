use if_chain::if_chain;
use rustc::lint::in_external_macro;
use rustc_hir::{PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::{is_must_use_func_call, is_must_use_ty, match_type, paths, span_lint_and_help};

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

declare_clippy_lint! {
    /// **What it does:** Checks for `let _ = sync_lock`
    ///
    /// **Why is this bad?** This statement immediately drops the lock instead of
    /// extending it's lifetime to the end of the scope, which is often not intended.
    /// To extend lock lifetime to the end of the scope, use an underscore-prefixed
    /// name instead (i.e. _lock). If you want to explicitly drop the lock,
    /// `std::mem::drop` conveys your intention better and is less error-prone.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// Bad:
    /// ```rust,ignore
    /// let _ = mutex.lock();
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// let _lock = mutex.lock();
    /// ```
    pub LET_UNDERSCORE_LOCK,
    correctness,
    "non-binding let on a synchronization lock"
}

declare_lint_pass!(LetUnderscore => [LET_UNDERSCORE_MUST_USE, LET_UNDERSCORE_LOCK]);

const SYNC_GUARD_PATHS: [&[&str]; 3] = [
    &paths::MUTEX_GUARD,
    &paths::RWLOCK_READ_GUARD,
    &paths::RWLOCK_WRITE_GUARD,
];

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
                let check_ty = |ty| SYNC_GUARD_PATHS.iter().any(|path| match_type(cx, ty, path));
                if cx.tables.expr_ty(init).walk().any(check_ty) {
                    span_lint_and_help(
                        cx,
                        LET_UNDERSCORE_LOCK,
                        stmt.span,
                        "non-binding let on a synchronization lock",
                        "consider using an underscore-prefixed named \
                            binding or dropping explicitly with `std::mem::drop`"
                    )
                } else if is_must_use_ty(cx, cx.tables.expr_ty(init)) {
                    span_lint_and_help(
                        cx,
                        LET_UNDERSCORE_MUST_USE,
                        stmt.span,
                        "non-binding let on an expression with `#[must_use]` type",
                        "consider explicitly using expression value"
                    )
                } else if is_must_use_func_call(cx, init) {
                    span_lint_and_help(
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

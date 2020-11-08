use if_chain::if_chain;
use rustc_hir::{Local, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::{implements_trait, is_must_use_func_call, is_must_use_ty, match_type, paths, span_lint_and_help};

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
    /// extending its lifetime to the end of the scope, which is often not intended.
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

declare_clippy_lint! {
    /// **What it does:** Checks for `let _ = <expr>`
    /// where expr has a type that implements `Drop`
    ///
    /// **Why is this bad?** This statement immediately drops the initializer
    /// expression instead of extending its lifetime to the end of the scope, which
    /// is often not intended. To extend the expression's lifetime to the end of the
    /// scope, use an underscore-prefixed name instead (i.e. _var). If you want to
    /// explicitly drop the expression, `std::mem::drop` conveys your intention
    /// better and is less error-prone.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// Bad:
    /// ```rust,ignore
    /// struct Droppable;
    /// impl Drop for Droppable {
    ///     fn drop(&mut self) {}
    /// }
    /// {
    ///     let _ = Droppable;
    ///     //               ^ dropped here
    ///     /* more code */
    /// }
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// {
    ///     let _droppable = Droppable;
    ///     /* more code */
    ///     // dropped at end of scope
    /// }
    /// ```
    pub LET_UNDERSCORE_DROP,
    pedantic,
    "non-binding let on a type that implements `Drop`"
}

declare_lint_pass!(LetUnderscore => [LET_UNDERSCORE_MUST_USE, LET_UNDERSCORE_LOCK, LET_UNDERSCORE_DROP]);

const SYNC_GUARD_PATHS: [&[&str]; 3] = [
    &paths::MUTEX_GUARD,
    &paths::RWLOCK_READ_GUARD,
    &paths::RWLOCK_WRITE_GUARD,
];

impl<'tcx> LateLintPass<'tcx> for LetUnderscore {
    fn check_local(&mut self, cx: &LateContext<'_>, local: &Local<'_>) {
        if in_external_macro(cx.tcx.sess, local.span) {
            return;
        }

        if_chain! {
            if let PatKind::Wild = local.pat.kind;
            if let Some(ref init) = local.init;
            then {
                let init_ty = cx.typeck_results().expr_ty(init);
                let contains_sync_guard = init_ty.walk().any(|inner| match inner.unpack() {
                    GenericArgKind::Type(inner_ty) => {
                        SYNC_GUARD_PATHS.iter().any(|path| match_type(cx, inner_ty, path))
                    },

                    GenericArgKind::Lifetime(_) | GenericArgKind::Const(_) => false,
                });
                let implements_drop = cx.tcx.lang_items().drop_trait().map_or(false, |drop_trait|
                    init_ty.walk().any(|inner| match inner.unpack() {
                        GenericArgKind::Type(inner_ty) => {
                            implements_trait(cx, inner_ty, drop_trait, &[])
                        },

                        GenericArgKind::Lifetime(_) | GenericArgKind::Const(_) => false,
                    })
                );
                if contains_sync_guard {
                    span_lint_and_help(
                        cx,
                        LET_UNDERSCORE_LOCK,
                        local.span,
                        "non-binding let on a synchronization lock",
                        None,
                        "consider using an underscore-prefixed named \
                            binding or dropping explicitly with `std::mem::drop`"
                    )
                } else if implements_drop {
                    span_lint_and_help(
                        cx,
                        LET_UNDERSCORE_DROP,
                        local.span,
                        "non-binding `let` on a type that implements `Drop`",
                        None,
                        "consider using an underscore-prefixed named \
                            binding or dropping explicitly with `std::mem::drop`"
                    )
                } else if is_must_use_ty(cx, cx.typeck_results().expr_ty(init)) {
                    span_lint_and_help(
                        cx,
                        LET_UNDERSCORE_MUST_USE,
                        local.span,
                        "non-binding let on an expression with `#[must_use]` type",
                        None,
                        "consider explicitly using expression value"
                    )
                } else if is_must_use_func_call(cx, init) {
                    span_lint_and_help(
                        cx,
                        LET_UNDERSCORE_MUST_USE,
                        local.span,
                        "non-binding let on a result of a `#[must_use]` function",
                        None,
                        "consider explicitly using function result"
                    )
                }
            }
        }
    }
}

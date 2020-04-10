use crate::utils::{match_def_path, paths, span_lint_and_note};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, HirId, IsAsync};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for calls to await while holding a MutexGuard.
    ///
    /// **Why is this bad?** This is almost certainly an error which can result
    /// in a deadlock because the reactor will invoke code not visible to the
    /// currently visible scope.
    ///
    /// **Known problems:** Detects only specifically named guard types:
    /// MutexGuard, RwLockReadGuard, and RwLockWriteGuard.
    ///
    /// **Example:**
    ///
    /// ```rust,ignore
    /// use std::sync::Mutex;
    ///
    /// async fn foo(x: &Mutex<u32>) {
    ///   let guard = x.lock().unwrap();
    ///   *guard += 1;
    ///   bar.await;
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// use std::sync::Mutex;
    ///
    /// async fn foo(x: &Mutex<u32>) {
    ///   {
    ///     let guard = x.lock().unwrap();
    ///     *guard += 1;
    ///   }
    ///   bar.await;
    /// }
    /// ```
    pub AWAIT_HOLDING_LOCK,
    pedantic,
    "Inside an async function, holding a MutexGuard while calling await"
}

declare_lint_pass!(AwaitHoldingLock => [AWAIT_HOLDING_LOCK]);

impl LateLintPass<'_, '_> for AwaitHoldingLock {
    fn check_fn(
        &mut self,
        cx: &LateContext<'_, '_>,
        fn_kind: FnKind<'_>,
        _: &FnDecl<'_>,
        _: &Body<'_>,
        span: Span,
        _: HirId,
    ) {
        if !is_async_fn(fn_kind) {
            return;
        }

        for ty_cause in &cx.tables.generator_interior_types {
            if let rustc_middle::ty::Adt(adt, _) = ty_cause.ty.kind {
                if is_mutex_guard(cx, adt.did) {
                    span_lint_and_note(
                        cx,
                        AWAIT_HOLDING_LOCK,
                        ty_cause.span,
                        "this MutexGuard is held across an 'await' point",
                        ty_cause.scope_span.unwrap_or(span),
                        "these are all the await points this lock is held through",
                    );
                }
            }
        }
    }
}

fn is_async_fn(fn_kind: FnKind<'_>) -> bool {
    fn_kind.header().map_or(false, |h| match h.asyncness {
        IsAsync::Async => true,
        IsAsync::NotAsync => false,
    })
}

fn is_mutex_guard(cx: &LateContext<'_, '_>, def_id: DefId) -> bool {
    match_def_path(cx, def_id, &paths::MUTEX_GUARD)
        || match_def_path(cx, def_id, &paths::RWLOCK_READ_GUARD)
        || match_def_path(cx, def_id, &paths::RWLOCK_WRITE_GUARD)
        || match_def_path(cx, def_id, &paths::PARKING_LOT_MUTEX_GUARD)
        || match_def_path(cx, def_id, &paths::PARKING_LOT_RWLOCK_READ_GUARD)
        || match_def_path(cx, def_id, &paths::PARKING_LOT_RWLOCK_WRITE_GUARD)
}

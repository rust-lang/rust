use crate::utils::span_lint_and_note;
use if_chain::if_chain;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, HirId, IsAsync};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{Span, Symbol};

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
    /// ```rust
    /// use std::sync::Mutex;
    ///
    /// async fn foo(x: &Mutex<u32>) {
    ///   let guard = x.lock().unwrap();
    ///   *guard += 1;
    ///   bar.await;
    /// }
    /// ```
    /// Use instead:
    /// ```rust
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

const MUTEX_GUARD_TYPES: [&str; 3] = ["MutexGuard", "RwLockReadGuard", "RwLockWriteGuard"];

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

        for ty_clause in &cx.tables.generator_interior_types {
            if_chain! {
              if let rustc_middle::ty::Adt(adt, _) = ty_clause.ty.kind;
              if let Some(&sym) = cx.get_def_path(adt.did).iter().last();
              if is_symbol_mutex_guard(sym);
              then {
                span_lint_and_note(
                      cx,
                      AWAIT_HOLDING_LOCK,
                      ty_clause.span,
                      "this MutexGuard is held across an 'await' point",
                      ty_clause.scope_span.unwrap_or(span),
                      "these are all the await points this lock is held through"
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

fn is_symbol_mutex_guard(sym: Symbol) -> bool {
    let sym_str = sym.as_str();
    for ty in &MUTEX_GUARD_TYPES {
        if sym_str == *ty {
            return true;
        }
    }
    false
}

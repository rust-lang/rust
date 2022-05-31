use crate::{LateContext, LateLintPass, LintContext};
use rustc_hir as hir;
use rustc_middle::ty::{self, subst::GenericArgKind};
use rustc_span::Symbol;

declare_lint! {
    /// The `let_underscore_drop` lint checks for statements which don't bind
    /// an expression which has a non-trivial Drop implementation to anything,
    /// causing the expression to be dropped immediately instead of at end of
    /// scope.
    ///
    /// ### Example
    /// ```rust
    /// struct SomeStruct;
    /// impl Drop for SomeStruct {
    ///     fn drop(&mut self) {
    ///         println!("Dropping SomeStruct");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // SomeStuct is dropped immediately instead of at end of scope,
    ///     // so "Dropping SomeStruct" is printed before "end of main".
    ///     // The order of prints would be reversed if SomeStruct was bound to
    ///     // a name (such as "_foo").
    ///     let _ = SomeStruct;
    ///     println!("end of main");
    /// }
    /// ```
    /// ### Explanation
    ///
    /// Statements which assign an expression to an underscore causes the
    /// expression to immediately drop instead of extending the expression's
    /// lifetime to the end of the scope. This is usually unintended,
    /// especially for types like `MutexGuard`, which are typically used to
    /// lock a mutex for the duration of an entire scope.
    ///
    /// If you want to extend the expression's lifetime to the end of the scope,
    /// assign an underscore-prefixed name (such as `_foo`) to the expression.
    /// If you do actually want to drop the expression immediately, then
    /// calling `std::mem::drop` on the expression is clearer and helps convey
    /// intent.
    pub LET_UNDERSCORE_DROP,
    Warn,
    "non-binding let on a type that implements `Drop`"
}

declare_lint! {
    /// The `let_underscore_lock` lint checks for statements which don't bind
    /// a mutex to anything, causing the lock to be released immediately instead
    /// of at end of scope, which is typically incorrect.
    ///
    /// ### Example
    /// ```rust
    /// use std::sync::{Arc, Mutex};
    /// use std::thread;
    /// let data = Arc::new(Mutex::new(0));
    ///
    /// thread::spawn(move || {
    ///     // The lock is immediately released instead of at the end of the
    ///     // scope, which is probably not intended.
    ///     let _ = data.lock().unwrap();
    ///     println!("doing some work");
    ///     let mut lock = data.lock().unwrap();
    ///     *lock += 1;
    /// });
    /// ```
    /// ### Explanation
    ///
    /// Statements which assign an expression to an underscore causes the
    /// expression to immediately drop instead of extending the expression's
    /// lifetime to the end of the scope. This is usually unintended,
    /// especially for types like `MutexGuard`, which are typically used to
    /// lock a mutex for the duration of an entire scope.
    ///
    /// If you want to extend the expression's lifetime to the end of the scope,
    /// assign an underscore-prefixed name (such as `_foo`) to the expression.
    /// If you do actually want to drop the expression immediately, then
    /// calling `std::mem::drop` on the expression is clearer and helps convey
    /// intent.
    pub LET_UNDERSCORE_LOCK,
    Warn,
    "non-binding let on a synchronization lock"
}

declare_lint_pass!(LetUnderscore => [LET_UNDERSCORE_DROP, LET_UNDERSCORE_LOCK]);

const SYNC_GUARD_PATHS: [&[&str]; 5] = [
    &["std", "sync", "mutex", "MutexGuard"],
    &["std", "sync", "rwlock", "RwLockReadGuard"],
    &["std", "sync", "rwlock", "RwLockWriteGuard"],
    &["parking_lot", "raw_mutex", "RawMutex"],
    &["parking_lot", "raw_rwlock", "RawRwLock"],
];

impl<'tcx> LateLintPass<'tcx> for LetUnderscore {
    fn check_local(&mut self, cx: &LateContext<'_>, local: &hir::Local<'_>) {
        if !matches!(local.pat.kind, hir::PatKind::Wild) {
            return;
        }
        if let Some(init) = local.init {
            let init_ty = cx.typeck_results().expr_ty(init);
            let needs_drop = init_ty.needs_drop(cx.tcx, cx.param_env);
            let is_sync_lock = init_ty.walk().any(|inner| match inner.unpack() {
                GenericArgKind::Type(inner_ty) => {
                    SYNC_GUARD_PATHS.iter().any(|guard_path| match inner_ty.kind() {
                        ty::Adt(adt, _) => {
                            let ty_path = cx.get_def_path(adt.did());
                            guard_path.iter().map(|x| Symbol::intern(x)).eq(ty_path.iter().copied())
                        }
                        _ => false,
                    })
                }

                GenericArgKind::Lifetime(_) | GenericArgKind::Const(_) => false,
            });
            if is_sync_lock {
                cx.struct_span_lint(LET_UNDERSCORE_LOCK, local.span, |lint| {
                    lint.build("non-binding let on a synchronization lock")
                        .help("consider binding to an unused variable")
                        .help("consider explicitly droping with `std::mem::drop`")
                        .emit();
                })
            } else if needs_drop {
                cx.struct_span_lint(LET_UNDERSCORE_DROP, local.span, |lint| {
                    lint.build("non-binding let on a type that implements `Drop`")
                        .help("consider binding to an unused variable")
                        .help("consider explicitly droping with `std::mem::drop`")
                        .emit();
                })
            }
        }
    }
}

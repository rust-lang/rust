use crate::{LateContext, LateLintPass, LintContext};
use rustc_errors::{Applicability, LintDiagnosticBuilder, MultiSpan};
use rustc_hir as hir;
use rustc_middle::ty;
use rustc_span::Symbol;

declare_lint! {
    /// The `let_underscore_drop` lint checks for statements which don't bind
    /// an expression which has a non-trivial Drop implementation to anything,
    /// causing the expression to be dropped immediately instead of at end of
    /// scope.
    ///
    /// ### Example
    /// ```
    /// struct SomeStruct;
    /// impl Drop for SomeStruct {
    ///     fn drop(&mut self) {
    ///         println!("Dropping SomeStruct");
    ///     }
    /// }
    ///
    /// fn main() {
    ///    #[warn(let_underscore_drop)]
    ///     // SomeStuct is dropped immediately instead of at end of scope,
    ///     // so "Dropping SomeStruct" is printed before "end of main".
    ///     // The order of prints would be reversed if SomeStruct was bound to
    ///     // a name (such as "_foo").
    ///     let _ = SomeStruct;
    ///     println!("end of main");
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
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
    Allow,
    "non-binding let on a type that implements `Drop`"
}

declare_lint! {
    /// The `let_underscore_lock` lint checks for statements which don't bind
    /// a mutex to anything, causing the lock to be released immediately instead
    /// of at end of scope, which is typically incorrect.
    ///
    /// ### Example
    /// ```compile_fail
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
    ///
    /// {{produces}}
    ///
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
    Deny,
    "non-binding let on a synchronization lock"
}

declare_lint_pass!(LetUnderscore => [LET_UNDERSCORE_DROP, LET_UNDERSCORE_LOCK]);

const SYNC_GUARD_SYMBOLS: [Symbol; 3] = [
    rustc_span::sym::MutexGuard,
    rustc_span::sym::RwLockReadGuard,
    rustc_span::sym::RwLockWriteGuard,
];

impl<'tcx> LateLintPass<'tcx> for LetUnderscore {
    fn check_local(&mut self, cx: &LateContext<'_>, local: &hir::Local<'_>) {
        if !matches!(local.pat.kind, hir::PatKind::Wild) {
            return;
        }
        if let Some(init) = local.init {
            let init_ty = cx.typeck_results().expr_ty(init);
            // If the type has a trivial Drop implementation, then it doesn't
            // matter that we drop the value immediately.
            if !init_ty.needs_drop(cx.tcx, cx.param_env) {
                return;
            }
            let is_sync_lock = match init_ty.kind() {
                ty::Adt(adt, _) => SYNC_GUARD_SYMBOLS
                    .iter()
                    .any(|guard_symbol| cx.tcx.is_diagnostic_item(*guard_symbol, adt.did())),
                _ => false,
            };

            if is_sync_lock {
                let mut span = MultiSpan::from_spans(vec![local.pat.span, init.span]);
                span.push_span_label(
                    local.pat.span,
                    "this lock is not assigned to a binding and is immediately dropped".to_string(),
                );
                span.push_span_label(
                    init.span,
                    "this binding will immediately drop the value assigned to it".to_string(),
                );
                cx.struct_span_lint(LET_UNDERSCORE_LOCK, span, |lint| {
                    build_and_emit_lint(
                        lint,
                        local,
                        init.span,
                        "non-binding let on a synchronization lock",
                    )
                })
            } else {
                cx.struct_span_lint(LET_UNDERSCORE_DROP, local.span, |lint| {
                    build_and_emit_lint(
                        lint,
                        local,
                        init.span,
                        "non-binding let on a type that implements `Drop`",
                    );
                })
            }
        }
    }
}

fn build_and_emit_lint(
    lint: LintDiagnosticBuilder<'_, ()>,
    local: &hir::Local<'_>,
    init_span: rustc_span::Span,
    msg: &str,
) {
    lint.build(msg)
        .span_suggestion_verbose(
            local.pat.span,
            "consider binding to an unused variable to avoid immediately dropping the value",
            "_unused",
            Applicability::MachineApplicable,
        )
        .multipart_suggestion(
            "consider immediately dropping the value",
            vec![
                (local.span.until(init_span), "drop(".to_string()),
                (init_span.shrink_to_hi(), ")".to_string()),
            ],
            Applicability::MachineApplicable,
        )
        .emit();
}

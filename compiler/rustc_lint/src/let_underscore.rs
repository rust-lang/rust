use crate::{LateContext, LateLintPass, LintContext};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_middle::{
    lint::LintDiagnosticBuilder,
    ty::{self, Ty},
};
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

declare_lint! {
    /// The `let_underscore_must_use` lint checks for statements which don't bind
    /// a `must_use` expression to anything, causing the lock to be released
    /// immediately instead of at end of scope, which is typically incorrect.
    ///
    /// ### Example
    /// ```rust
    /// #[must_use]
    /// struct SomeStruct;
    ///
    /// fn main() {
    ///     // SomeStuct is dropped immediately instead of at end of scope.
    ///     let _ = SomeStruct;
    /// }
    /// ```
    /// ### Explanation
    ///
    /// Statements which assign an expression to an underscore causes the
    /// expression to immediately drop. Usually, it's better to explicitly handle
    /// the `must_use` expression.
    pub LET_UNDERSCORE_MUST_USE,
    Allow,
    "non-binding let on a expression marked `must_use`"
}

declare_lint_pass!(LetUnderscore => [LET_UNDERSCORE_DROP, LET_UNDERSCORE_LOCK, LET_UNDERSCORE_MUST_USE]);

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
            let is_must_use_ty = is_must_use_ty(cx, cx.typeck_results().expr_ty(init));
            let is_must_use_func_call = is_must_use_func_call(cx, init);

            if is_sync_lock {
                cx.struct_span_lint(LET_UNDERSCORE_LOCK, local.span, |lint| {
                    build_and_emit_lint(
                        lint,
                        local,
                        init.span,
                        "non-binding let on a synchronization lock",
                    )
                })
            } else if is_must_use_ty || is_must_use_func_call {
                cx.struct_span_lint(LET_UNDERSCORE_MUST_USE, local.span, |lint| {
                    build_and_emit_lint(
                        lint,
                        local,
                        init.span,
                        "non-binding let on a expression marked `must_use`",
                    );
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
            "consider binding to an unused variable",
            "_unused",
            Applicability::MachineApplicable,
        )
        .span_suggestion_verbose(
            init_span,
            "consider explicitly droping with `std::mem::drop`",
            "drop(...)",
            Applicability::HasPlaceholders,
        )
        .emit();
}

// return true if `ty` is a type that is marked as `must_use`
fn is_must_use_ty<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        ty::Adt(adt, _) => has_must_use_attr(cx, adt.did()),
        ty::Foreign(ref did) => has_must_use_attr(cx, *did),
        ty::Slice(ty)
        | ty::Array(ty, _)
        | ty::RawPtr(ty::TypeAndMut { ty, .. })
        | ty::Ref(_, ty, _) => {
            // for the Array case we don't need to care for the len == 0 case
            // because we don't want to lint functions returning empty arrays
            is_must_use_ty(cx, *ty)
        }
        ty::Tuple(substs) => substs.iter().any(|ty| is_must_use_ty(cx, ty)),
        ty::Opaque(ref def_id, _) => {
            for (predicate, _) in cx.tcx.explicit_item_bounds(*def_id) {
                if let ty::PredicateKind::Trait(trait_predicate) = predicate.kind().skip_binder() {
                    if has_must_use_attr(cx, trait_predicate.trait_ref.def_id) {
                        return true;
                    }
                }
            }
            false
        }
        ty::Dynamic(binder, _) => {
            for predicate in binder.iter() {
                if let ty::ExistentialPredicate::Trait(ref trait_ref) = predicate.skip_binder() {
                    if has_must_use_attr(cx, trait_ref.def_id) {
                        return true;
                    }
                }
            }
            false
        }
        _ => false,
    }
}

// check if expr is calling method or function with #[must_use] attribute
fn is_must_use_func_call(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    let did = match expr.kind {
                hir::ExprKind::Call(path, _) if let hir::ExprKind::Path(ref qpath) = path.kind => {
                    if let hir::def::Res::Def(_, did) = cx.qpath_res(qpath, path.hir_id) {
                        Some(did)
                    } else {
                        None
                    }
                },
                hir::ExprKind::MethodCall(..) => {
                    cx.typeck_results().type_dependent_def_id(expr.hir_id)
                }
                _ => None,
            };

    did.map_or(false, |did| has_must_use_attr(cx, did))
}

// returns true if DefId contains a `#[must_use]` attribute
fn has_must_use_attr(cx: &LateContext<'_>, did: hir::def_id::DefId) -> bool {
    cx.tcx.has_attr(did, rustc_span::sym::must_use)
}

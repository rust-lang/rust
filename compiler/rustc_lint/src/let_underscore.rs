use rustc_errors::MultiSpan;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_lint_defs::{declare_lint, declare_lint_pass};
use rustc_middle::ty;
use rustc_span::{Symbol, sym};

use crate::diagnostics::{NonBindingLet, NonBindingLetSub};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `let_underscore_drop` lint checks for statements which don't bind
    /// an expression which has a non-trivial Drop implementation to anything,
    /// causing the expression to be dropped immediately instead of at end of
    /// scope.
    ///
    /// ### Example
    ///
    /// ```rust
    /// struct SomeStruct;
    /// impl Drop for SomeStruct {
    ///     fn drop(&mut self) {
    ///         println!("Dropping SomeStruct");
    ///     }
    /// }
    ///
    /// fn main() {
    ///    #[warn(let_underscore_drop)]
    ///     // SomeStruct is dropped immediately instead of at end of scope,
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
    "non-binding let on a type that has a destructor"
}

declare_lint! {
    /// The `let_underscore_lock` lint checks for statements which don't bind
    /// a mutex to anything, causing the lock to be released immediately instead
    /// of at end of scope, which is typically incorrect.
    ///
    /// ### Example
    /// ```rust,compile_fail
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
    fn check_local(&mut self, cx: &LateContext<'_>, local: &hir::LetStmt<'_>) {
        if matches!(local.source, rustc_hir::LocalSource::AsyncFn) {
            return;
        }

        let mut top_level = true;

        // We recursively walk through all patterns, so that we can catch cases where the lock is
        // nested in a pattern. For the basic `let_underscore_drop` lint, we only look at the top
        // level, since there are many legitimate reasons to bind a sub-pattern to an `_`, if we're
        // only interested in the rest. But with locks, we prefer having the chance of "false
        // positives" over missing cases, since the effects can be quite catastrophic.
        local.pat.walk_always(|pat| {
            let is_top_level = top_level;
            top_level = false;

            if !matches!(pat.kind, hir::PatKind::Wild) {
                return;
            }

            // `let _ = place;` doesn't move out of the place, so nothing gets dropped.
            if let Some(init) = local.init
                && cx.typeck_results().expr_adjustments(init).is_empty()
                && (is_local_or_static_place(init)
                    || borrows_extended_temporary(cx, init, local.pat))
            {
                return;
            }

            let ty = cx.typeck_results().pat_ty(pat);

            // If the type has a trivial Drop implementation, then it doesn't
            // matter that we drop the value immediately.
            if !ty.needs_drop(cx.tcx, cx.typing_env()) {
                return;
            }
            // Lint for patterns like `mutex.lock()`, which returns `Result<MutexGuard, _>` as well.
            let potential_lock_type = match ty.kind() {
                ty::Adt(adt, args) if cx.tcx.is_diagnostic_item(sym::Result, adt.did()) => {
                    args.type_at(0)
                }
                _ => ty,
            };
            let is_sync_lock = match potential_lock_type.kind() {
                ty::Adt(adt, _) => SYNC_GUARD_SYMBOLS
                    .iter()
                    .any(|guard_symbol| cx.tcx.is_diagnostic_item(*guard_symbol, adt.did())),
                _ => false,
            };

            let can_use_init = is_top_level.then_some(local.init).flatten();

            let sub = NonBindingLetSub {
                suggestion: pat.span,
                // We can't suggest `drop()` when we're on the top level.
                drop_fn_start_end: can_use_init
                    .map(|init| (local.span.until(init.span), init.span.shrink_to_hi())),
                is_assign_desugar: matches!(local.source, rustc_hir::LocalSource::AssignDesugar),
            };
            if is_sync_lock {
                let span = MultiSpan::from_span(pat.span);
                cx.emit_span_lint(
                    LET_UNDERSCORE_LOCK,
                    span,
                    NonBindingLet::SyncLock { sub, pat: pat.span },
                );
            // Only emit let_underscore_drop for top-level `_` patterns.
            } else if can_use_init.is_some() {
                cx.emit_span_lint(LET_UNDERSCORE_DROP, local.span, NonBindingLet::DropType { sub });
            }
        });
    }
}

/// Whether `expr` is a place based on a local or a static, so no temporary is involved.
// FIXME: this doesn't look through calls, so `let _ = *identity(&x);` still lints even though
// nothing is dropped. Telling that apart from `*identity(&temp())` needs more than the HIR.
fn is_local_or_static_place(expr: &hir::Expr<'_>) -> bool {
    match expr.kind {
        hir::ExprKind::Path(hir::QPath::Resolved(_, path)) => {
            matches!(path.res, Res::Local(_) | Res::Def(DefKind::Static { .. }, _))
        }
        hir::ExprKind::Field(base, _)
        | hir::ExprKind::Index(base, _, _)
        | hir::ExprKind::Unary(hir::UnOp::Deref, base)
        | hir::ExprKind::AddrOf(_, _, base)
        | hir::ExprKind::Type(base, _) => is_local_or_static_place(base),
        _ => false,
    }
}

/// Whether `expr` borrows a temporary whose lifetime is extended past the `let`, like
/// `let &_ = &m.lock().unwrap();`, and `pat` doesn't look behind any other reference.
fn borrows_extended_temporary(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    pat: &hir::Pat<'_>,
) -> bool {
    let Some(body) = cx.enclosing_body else { return false };
    let mut root = expr;
    let mut borrows = 0;
    loop {
        if !cx.typeck_results().expr_adjustments(root).is_empty() {
            return false;
        }
        match root.kind {
            hir::ExprKind::AddrOf(_, _, base) => {
                borrows += 1;
                root = base;
            }
            hir::ExprKind::Field(base, _) => root = base,
            // Only temporaries get their lifetime extended, places and paths don't.
            hir::ExprKind::Unary(hir::UnOp::Deref, _)
            | hir::ExprKind::Index(..)
            | hir::ExprKind::Type(..)
            | hir::ExprKind::Path(_) => return false,
            _ => break,
        }
    }

    // Each deref in the pattern has to go through one of the borrows above, otherwise it could
    // reach a temporary that isn't extended, like in `let &&_ = &identity(&m.lock());`.
    let mut derefs = 0;
    let mut deref_pat = false;
    pat.walk_always(|p| {
        derefs += cx.typeck_results().pat_adjustments().get(p.hir_id).map_or(0, |a| a.len());
        match p.kind {
            hir::PatKind::Ref(..) => derefs += 1,
            hir::PatKind::Deref(_) => deref_pat = true,
            _ => {}
        }
    });
    if deref_pat || derefs > borrows {
        return false;
    }

    let scope_tree = cx.tcx.region_scope_tree(cx.tcx.hir_body_owner_def_id(body));
    scope_tree.temporary_scope(root.hir_id.local_id).temp_lifetime
        != scope_tree.temporary_scope(expr.hir_id.local_id).temp_lifetime
}

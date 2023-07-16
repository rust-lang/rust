use crate::context::LintContext;
use crate::lints::{
    NoopMethodCallDiag, SuspiciousDoubleRefCloneDiag, SuspiciousDoubleRefDerefDiag,
};
use crate::LateContext;
use crate::LateLintPass;
use rustc_hir::def::DefKind;
use rustc_hir::{Expr, ExprKind};
use rustc_middle::ty;
use rustc_middle::ty::adjustment::Adjust;
use rustc_span::symbol::sym;

declare_lint! {
    /// The `noop_method_call` lint detects specific calls to noop methods
    /// such as a calling `<&T as Clone>::clone` where `T: !Clone`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// #![warn(noop_method_call)]
    /// struct Foo;
    /// let foo = &Foo;
    /// let clone: &Foo = foo.clone();
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Some method calls are noops meaning that they do nothing. Usually such methods
    /// are the result of blanket implementations that happen to create some method invocations
    /// that end up not doing anything. For instance, `Clone` is implemented on all `&T`, but
    /// calling `clone` on a `&T` where `T` does not implement clone, actually doesn't do anything
    /// as references are copy. This lint detects these calls and warns the user about them.
    pub NOOP_METHOD_CALL,
    Allow,
    "detects the use of well-known noop methods"
}

declare_lint! {
    /// The `suspicious_double_ref_op` lint checks for usage of `.clone()`/`.borrow()`/`.deref()`
    /// on an `&&T` when `T: !Deref/Borrow/Clone`, which means the call will return the inner `&T`,
    /// instead of performing the operation on the underlying `T` and can be confusing.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// struct Foo;
    /// let foo = &&Foo;
    /// let clone: &Foo = foo.clone();
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Since `Foo` doesn't implement `Clone`, running `.clone()` only dereferences the double
    /// reference, instead of cloning the inner type which should be what was intended.
    pub SUSPICIOUS_DOUBLE_REF_OP,
    Warn,
    "suspicious call of trait method on `&&T`"
}

declare_lint_pass!(NoopMethodCall => [NOOP_METHOD_CALL, SUSPICIOUS_DOUBLE_REF_OP]);

impl<'tcx> LateLintPass<'tcx> for NoopMethodCall {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // We only care about method calls.
        let ExprKind::MethodCall(call, receiver, _, call_span) = &expr.kind else {
            return;
        };

        if call_span.from_expansion() {
            return;
        }

        // We only care about method calls corresponding to the `Clone`, `Deref` and `Borrow`
        // traits and ignore any other method call.

        let Some((DefKind::AssocFn, did)) = cx.typeck_results().type_dependent_def(expr.hir_id)
        else {
            return;
        };

        let Some(trait_id) = cx.tcx.trait_of_item(did) else { return };

        if !matches!(
            cx.tcx.get_diagnostic_name(trait_id),
            Some(sym::Borrow | sym::Clone | sym::Deref)
        ) {
            return;
        };

        let args = cx
            .tcx
            .normalize_erasing_regions(cx.param_env, cx.typeck_results().node_args(expr.hir_id));
        // Resolve the trait method instance.
        let Ok(Some(i)) = ty::Instance::resolve(cx.tcx, cx.param_env, did, args) else { return };
        // (Re)check that it implements the noop diagnostic.
        let Some(name) = cx.tcx.get_diagnostic_name(i.def_id()) else { return };

        let receiver_ty = cx.typeck_results().expr_ty(receiver);
        let expr_ty = cx.typeck_results().expr_ty_adjusted(expr);
        let arg_adjustments = cx.typeck_results().expr_adjustments(receiver);

        // If there is any user defined auto-deref step, then we don't want to warn.
        // https://github.com/rust-lang/rust-clippy/issues/9272
        if arg_adjustments.iter().any(|adj| matches!(adj.kind, Adjust::Deref(Some(_)))) {
            return;
        }

        let expr_span = expr.span;
        let span = expr_span.with_lo(receiver.span.hi());

        if receiver_ty == expr_ty {
            cx.emit_spanned_lint(
                NOOP_METHOD_CALL,
                span,
                NoopMethodCallDiag { method: call.ident.name, receiver_ty, label: span },
            );
        } else {
            match name {
                // If `type_of(x) == T` and `x.borrow()` is used to get `&T`,
                // then that should be allowed
                sym::noop_method_borrow => return,
                sym::noop_method_clone => cx.emit_spanned_lint(
                    SUSPICIOUS_DOUBLE_REF_OP,
                    span,
                    SuspiciousDoubleRefCloneDiag { ty: expr_ty },
                ),
                sym::noop_method_deref => cx.emit_spanned_lint(
                    SUSPICIOUS_DOUBLE_REF_OP,
                    span,
                    SuspiciousDoubleRefDerefDiag { ty: expr_ty },
                ),
                _ => return,
            }
        }
    }
}

use crate::context::LintContext;
use crate::rustc_middle::ty::TypeFoldable;
use crate::LateContext;
use crate::LateLintPass;
use rustc_hir::def::DefKind;
use rustc_hir::{Expr, ExprKind};
use rustc_middle::ty;
use rustc_span::symbol::sym;

declare_lint! {
    /// The `noop_method_call` lint detects specific calls to noop methods
    /// such as a calling `<&T as Clone>::clone` where `T: !Clone`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
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
    Warn,
    "detects the use of well-known noop methods"
}

declare_lint_pass!(NoopMethodCall => [NOOP_METHOD_CALL]);

impl<'tcx> LateLintPass<'tcx> for NoopMethodCall {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // We only care about method calls
        if let ExprKind::MethodCall(..) = expr.kind {
            // Get the `DefId` only when dealing with an `AssocFn`
            if let Some((DefKind::AssocFn, did)) =
                cx.typeck_results().type_dependent_def(expr.hir_id)
            {
                // Check that we're dealing with a trait method
                if let Some(trait_id) = cx.tcx.trait_of_item(did) {
                    // Check we're dealing with one of the traits we care about
                    if ![sym::Clone, sym::Deref, sym::Borrow]
                        .iter()
                        .any(|s| cx.tcx.is_diagnostic_item(*s, trait_id))
                    {
                        return;
                    }

                    let substs = cx.typeck_results().node_substs(expr.hir_id);
                    // We can't resolve on types that recursively require monomorphization,
                    // so check that we don't need to perfom substitution
                    if !substs.needs_subst() {
                        let param_env = cx.tcx.param_env(trait_id);
                        // Resolve the trait method instance
                        if let Ok(Some(i)) = ty::Instance::resolve(cx.tcx, param_env, did, substs) {
                            // Check that it implements the noop diagnostic
                            if [
                                sym::noop_method_borrow,
                                sym::noop_method_clone,
                                sym::noop_method_deref,
                            ]
                            .iter()
                            .any(|s| cx.tcx.is_diagnostic_item(*s, i.def_id()))
                            {
                                let span = expr.span;

                                cx.struct_span_lint(NOOP_METHOD_CALL, span, |lint| {
                                    let message = "call to noop method";
                                    lint.build(&message).emit()
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

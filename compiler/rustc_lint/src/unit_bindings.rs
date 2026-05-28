use rustc_hir as hir;
use rustc_session::{declare_lint, declare_lint_pass};

use crate::lints::UnitBindingsDiag;
use crate::{LateLintPass, LintContext};

declare_lint! {
    /// The `unit_bindings` lint detects cases where bindings are useless because they have
    /// the unit type `()` as their inferred type. The lint is suppressed if the user explicitly
    /// annotates the let binding with the unit type `()`, or if the let binding uses an underscore
    /// wildcard pattern, i.e. `let _ = expr`, or if the binding is produced from macro expansions.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unit_bindings)]
    ///
    /// fn foo() {
    ///     println!("do work");
    /// }
    ///
    /// pub fn main() {
    ///     let x = foo(); // useless binding
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Creating a local binding with the unit type `()` does not do much and can be a sign of a
    /// user error, such as in this example:
    ///
    /// ```rust,no_run
    /// fn main() {
    ///     let mut x = [1, 2, 3];
    ///     x[0] = 5;
    ///     let y = x.sort(); // useless binding as `sort` returns `()` and not the sorted array.
    ///     println!("{:?}", y); // prints "()"
    /// }
    /// ```
    pub UNIT_BINDINGS,
    Allow,
    "binding is useless because it has the unit `()` type"
}

declare_lint_pass!(UnitBindings => [UNIT_BINDINGS]);

impl<'tcx> LateLintPass<'tcx> for UnitBindings {
    fn check_local(&mut self, cx: &crate::LateContext<'tcx>, local: &'tcx hir::LetStmt<'tcx>) {
        // Suppress warning if user:
        // - explicitly ascribes a type to the pattern
        // - explicitly wrote `let pat = ();`
        // - explicitly wrote `let () = init;`.
        if !local.span.from_expansion()
            && let Some(tyck_results) = cx.maybe_typeck_results()
            && let Some(init) = local.init
            && let init_ty = tyck_results.expr_ty(init)
            && let local_ty = tyck_results.node_type(local.hir_id)
            && init_ty == cx.tcx.types.unit
            && local_ty == cx.tcx.types.unit
            && local.ty.is_none()
            && !matches!(init.kind, hir::ExprKind::Tup([]))
            && !matches!(local.pat.kind, hir::PatKind::Tuple([], ..))
        {
            cx.emit_span_lint(
                UNIT_BINDINGS,
                local.span,
                UnitBindingsDiag { label: local.pat.span },
            );
        }
    }
}

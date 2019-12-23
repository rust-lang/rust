use crate::lint::{LateContext, LateLintPass, LintArray, LintContext, LintPass};
use rustc::{
    hir,
    lint::FutureIncompatibleInfo,
    ty::{
        self,
        adjustment::{Adjust, Adjustment},
    },
};
use syntax::{errors::Applicability, symbol::sym};

declare_lint! {
    pub ARRAY_INTO_ITER,
    Warn,
    "detects calling `into_iter` on arrays",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #66145 <https://github.com/rust-lang/rust/issues/66145>",
        edition: None,
    };
}

declare_lint_pass!(
    /// Checks for instances of calling `into_iter` on arrays.
    ArrayIntoIter => [ARRAY_INTO_ITER]
);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ArrayIntoIter {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        // We only care about method call expressions.
        if let hir::ExprKind::MethodCall(call, span, args) = &expr.kind {
            if call.ident.name != sym::into_iter {
                return;
            }

            // Check if the method call actually calls the libcore
            // `IntoIterator::into_iter`.
            let def_id = cx.tables.type_dependent_def_id(expr.hir_id).unwrap();
            match cx.tcx.trait_of_item(def_id) {
                Some(trait_id) if cx.tcx.is_diagnostic_item(sym::IntoIterator, trait_id) => {}
                _ => return,
            };

            // As this is a method call expression, we have at least one
            // argument.
            let receiver_arg = &args[0];

            // Test if the original `self` type is an array type.
            match cx.tables.expr_ty(receiver_arg).kind {
                ty::Array(..) => {}
                _ => return,
            }

            // Make sure that the first adjustment is an autoref coercion.
            match cx.tables.expr_adjustments(receiver_arg).get(0) {
                Some(Adjustment { kind: Adjust::Borrow(_), .. }) => {}
                _ => return,
            }

            // Emit lint diagnostic.
            let target = match cx.tables.expr_ty_adjusted(receiver_arg).kind {
                ty::Ref(_, ty::TyS { kind: ty::Array(..), .. }, _) => "[T; N]",
                ty::Ref(_, ty::TyS { kind: ty::Slice(..), .. }, _) => "[T]",

                // We know the original first argument type is an array type,
                // we know that the first adjustment was an autoref coercion
                // and we know that `IntoIterator` is the trait involved. The
                // array cannot be coerced to something other than a reference
                // to an array or to a slice.
                _ => bug!("array type coerced to something other than array or slice"),
            };
            let msg = format!(
                "this method call currently resolves to `<&{} as IntoIterator>::into_iter` (due \
                    to autoref coercions), but that might change in the future when \
                    `IntoIterator` impls for arrays are added.",
                target,
            );
            cx.struct_span_lint(ARRAY_INTO_ITER, *span, &msg)
                .span_suggestion(
                    call.ident.span,
                    "use `.iter()` instead of `.into_iter()` to avoid ambiguity",
                    "iter".into(),
                    Applicability::MachineApplicable,
                )
                .emit();
        }
    }
}

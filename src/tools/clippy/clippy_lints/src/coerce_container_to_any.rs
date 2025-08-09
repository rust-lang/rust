use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::{self, Sugg};
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::adjustment::{Adjust, PointerCoercion};
use rustc_middle::ty::{self, ExistentialPredicate, Ty, TyCtxt};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Protects against unintended coercion of references to container types to `&dyn Any` when the
    /// container type dereferences to a `dyn Any` which could be directly referenced instead.
    ///
    /// ### Why is this bad?
    ///
    /// The intention is usually to get a reference to the `dyn Any` the value dereferences to,
    /// rather than coercing a reference to the container itself to `&dyn Any`.
    ///
    /// ### Example
    ///
    /// Because `Box<dyn Any>` itself implements `Any`, `&Box<dyn Any>`
    /// can be coerced to an `&dyn Any` which refers to *the `Box` itself*, rather than the
    /// inner `dyn Any`.
    /// ```no_run
    /// # use std::any::Any;
    /// let x: Box<dyn Any> = Box::new(0u32);
    /// let dyn_any_of_box: &dyn Any = &x;
    ///
    /// // Fails as we have a &dyn Any to the Box, not the u32
    /// assert_eq!(dyn_any_of_box.downcast_ref::<u32>(), None);
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::any::Any;
    /// let x: Box<dyn Any> = Box::new(0u32);
    /// let dyn_any_of_u32: &dyn Any = &*x;
    ///
    /// // Succeeds since we have a &dyn Any to the inner u32!
    /// assert_eq!(dyn_any_of_u32.downcast_ref::<u32>(), Some(&0u32));
    /// ```
    #[clippy::version = "1.89.0"]
    pub COERCE_CONTAINER_TO_ANY,
    nursery,
    "coercing to `&dyn Any` when dereferencing could produce a `dyn Any` without coercion is usually not intended"
}
declare_lint_pass!(CoerceContainerToAny => [COERCE_CONTAINER_TO_ANY]);

impl<'tcx> LateLintPass<'tcx> for CoerceContainerToAny {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        // If this expression was coerced to `&dyn Any` ...
        if !cx.typeck_results().expr_adjustments(e).last().is_some_and(|adj| {
            matches!(adj.kind, Adjust::Pointer(PointerCoercion::Unsize)) && is_ref_dyn_any(cx.tcx, adj.target)
        }) {
            return;
        }

        let expr_ty = cx.typeck_results().expr_ty(e);
        let ty::Ref(_, expr_ref_ty, _) = *expr_ty.kind() else {
            return;
        };
        // ... but it's not actually `&dyn Any` ...
        if is_dyn_any(cx.tcx, expr_ref_ty) {
            return;
        }
        // ... and it also *derefs* to `dyn Any` ...
        let Some((depth, target)) = clippy_utils::ty::deref_chain(cx, expr_ref_ty).enumerate().last() else {
            return;
        };
        if !is_dyn_any(cx.tcx, target) {
            return;
        }

        // ... that's probably not intended.
        let (target_expr, deref_count) = match e.kind {
            // If `e` was already an `&` expression, skip `*&` in the suggestion
            ExprKind::AddrOf(_, _, referent) => (referent, depth),
            _ => (e, depth + 1),
        };
        let ty::Ref(_, _, mutability) = *cx.typeck_results().expr_ty_adjusted(e).kind() else {
            return;
        };
        let sugg = sugg::make_unop(
            &format!("{}{}", mutability.ref_prefix_str(), str::repeat("*", deref_count)),
            Sugg::hir(cx, target_expr, ".."),
        );
        span_lint_and_sugg(
            cx,
            COERCE_CONTAINER_TO_ANY,
            e.span,
            format!("coercing `{expr_ty}` to `{}dyn Any`", mutability.ref_prefix_str()),
            "consider dereferencing",
            sugg.to_string(),
            Applicability::MaybeIncorrect,
        );
    }
}

fn is_ref_dyn_any(tcx: TyCtxt<'_>, ty: Ty<'_>) -> bool {
    let ty::Ref(_, ref_ty, _) = *ty.kind() else {
        return false;
    };
    is_dyn_any(tcx, ref_ty)
}

fn is_dyn_any(tcx: TyCtxt<'_>, ty: Ty<'_>) -> bool {
    let ty::Dynamic(traits, ..) = ty.kind() else {
        return false;
    };
    traits.iter().any(|binder| {
        let ExistentialPredicate::Trait(t) = binder.skip_binder() else {
            return false;
        };
        tcx.is_diagnostic_item(sym::Any, t.def_id)
    })
}

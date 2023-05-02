use rustc_span::sym;

use crate::hir;

use crate::{lints::UnnecessarySendConstraintDiag, LateContext, LateLintPass};

declare_lint! {
    /// The `lint_unnecessary_send_constraint` lints unnecessary constraint of references to `Send`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// fn foo(_: &(dyn Any + Send>) {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// References cannot be sent across threads unless they have a `Sync` bound, so constraining them to `Send` without `Sync` is unnecessary.
    pub UNNECESSARY_SEND_CONSTRAINT,
    Warn,
    "constraining a reference to `Send` without `Sync` is unnecessary, consider removing it"
}

declare_lint_pass!(UnnecessarySendConstraint => [UNNECESSARY_SEND_CONSTRAINT]);

impl<'tcx> LateLintPass<'tcx> for UnnecessarySendConstraint {
    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'tcx>) {
        let hir::TyKind::Ref(
            ..,
            hir::MutTy {
                ty: hir::Ty {
                    kind: hir::TyKind::TraitObject(bounds, ..),
                    ..
                },
                ..
            },
        ) = ty.kind else { return; };

        let send = cx.tcx.get_diagnostic_item(sym::Send);

        let send_bound = bounds.iter().find(|b| b.trait_ref.trait_def_id() == send);

        if let Some(send_bound) = send_bound {
            let only_trait = bounds.len() == 1;

            cx.tcx.emit_spanned_lint(
                UNNECESSARY_SEND_CONSTRAINT,
                send_bound.trait_ref.hir_ref_id, // is this correct?
                send_bound.span,
                UnnecessarySendConstraintDiag { only_trait, suggestion: send_bound.span },
            )
        }
    }
}

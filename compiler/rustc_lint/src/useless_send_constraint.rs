use rustc_span::sym;

use crate::hir;

use crate::{lints::UselessSendConstraintDiag, LateContext, LateLintPass};

declare_lint! {
    /// The `lint_useless_send_constraint` lints useless constraint of references to `Send`.
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
    /// References cannot be sent across threads, so constraining them to `Send` is useless.
    pub USELESS_SEND_CONSTRAINT,
    Warn,
    "constraining a reference to `Send` is useless, consider removing it"
}

declare_lint_pass!(UselessSendConstraint => [USELESS_SEND_CONSTRAINT]);

impl<'tcx> LateLintPass<'tcx> for UselessSendConstraint {
    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'tcx>) {
        let hir::TyKind::Ref(
            _,
            hir::MutTy {
                ty: hir::Ty {
                    kind: hir::TyKind::TraitObject(bounds, ..),
                    ..
                },
                ..
            }
        ) = ty.kind else { return; };

        let send = cx.tcx.get_diagnostic_item(sym::Send);

        let send_bound = bounds.iter().find(|b| b.trait_ref.trait_def_id() == send);

        if let Some(send_bound) = send_bound {
            let only_trait = bounds.len() == 1;

            cx.tcx.emit_spanned_lint(
                USELESS_SEND_CONSTRAINT,
                send_bound.trait_ref.hir_ref_id, // is this correct?
                send_bound.span,
                UselessSendConstraintDiag { only_trait, suggestion: send_bound.span },
            )
        }
    }
}

use rustc_hir as hir;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{
    self, fold::BottomUpFolder, print::TraitPredPrintModifiersAndPath, Ty, TypeFoldable,
};
use rustc_span::Span;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;

use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `opaque_hidden_inferred_bound` lint detects cases in which nested
    /// `impl Trait` in associated type bounds are not written generally enough
    /// to satisfy the bounds of the associated type.
    ///
    /// ### Explanation
    ///
    /// This functionality was removed in #97346, but then rolled back in #99860
    /// because it caused regressions.
    ///
    /// We plan on reintroducing this as a hard error, but in the mean time,
    /// this lint serves to warn and suggest fixes for any use-cases which rely
    /// on this behavior.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(type_alias_impl_trait)]
    ///
    /// trait Duh {}
    ///
    /// impl Duh for i32 {}
    ///
    /// trait Trait {
    ///     type Assoc: Duh;
    /// }
    ///
    /// struct Struct;
    ///
    /// impl<F: Duh> Trait for F {
    ///     type Assoc = F;
    /// }
    ///
    /// type Tait = impl Sized;
    ///
    /// fn test() -> impl Trait<Assoc = Tait> {
    ///     42
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// In this example, `test` declares that the associated type `Assoc` for
    /// `impl Trait` is `impl Sized`, which does not satisfy the `Send` bound
    /// on the associated type.
    ///
    /// Although the hidden type, `i32` does satisfy this bound, we do not
    /// consider the return type to be well-formed with this lint. It can be
    /// fixed by changing `Tait = impl Sized` into `Tait = impl Sized + Send`.
    pub OPAQUE_HIDDEN_INFERRED_BOUND,
    Warn,
    "detects the use of nested `impl Trait` types in associated type bounds that are not general enough"
}

declare_lint_pass!(OpaqueHiddenInferredBound => [OPAQUE_HIDDEN_INFERRED_BOUND]);

impl<'tcx> LateLintPass<'tcx> for OpaqueHiddenInferredBound {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let hir::ItemKind::OpaqueTy(opaque) = &item.kind else { return; };
        let def_id = item.owner_id.def_id.to_def_id();
        let infcx = &cx.tcx.infer_ctxt().build();
        // For every projection predicate in the opaque type's explicit bounds,
        // check that the type that we're assigning actually satisfies the bounds
        // of the associated type.
        for (pred, pred_span) in cx.tcx.explicit_item_bounds(def_id).subst_identity_iter_copied() {
            // Liberate bound regions in the predicate since we
            // don't actually care about lifetimes in this check.
            let predicate = cx.tcx.liberate_late_bound_regions(def_id, pred.kind());
            let ty::ClauseKind::Projection(proj) = predicate else {
                continue;
            };
            // Only check types, since those are the only things that may
            // have opaques in them anyways.
            let Some(proj_term) = proj.term.ty() else { continue };

            // HACK: `impl Trait<Assoc = impl Trait2>` from an RPIT is "ok"...
            if let ty::Alias(ty::Opaque, opaque_ty) = *proj_term.kind()
                && cx.tcx.parent(opaque_ty.def_id) == def_id
                && matches!(
                    opaque.origin,
                    hir::OpaqueTyOrigin::FnReturn(_) | hir::OpaqueTyOrigin::AsyncFn(_)
                )
            {
                continue;
            }

            let proj_ty =
                cx.tcx.mk_projection(proj.projection_ty.def_id, proj.projection_ty.substs);
            // For every instance of the projection type in the bounds,
            // replace them with the term we're assigning to the associated
            // type in our opaque type.
            let proj_replacer = &mut BottomUpFolder {
                tcx: cx.tcx,
                ty_op: |ty| if ty == proj_ty { proj_term } else { ty },
                lt_op: |lt| lt,
                ct_op: |ct| ct,
            };
            // For example, in `impl Trait<Assoc = impl Send>`, for all of the bounds on `Assoc`,
            // e.g. `type Assoc: OtherTrait`, replace `<impl Trait as Trait>::Assoc: OtherTrait`
            // with `impl Send: OtherTrait`.
            for (assoc_pred, assoc_pred_span) in cx
                .tcx
                .explicit_item_bounds(proj.projection_ty.def_id)
                .subst_iter_copied(cx.tcx, &proj.projection_ty.substs)
            {
                let assoc_pred = assoc_pred.fold_with(proj_replacer);
                let Ok(assoc_pred) = traits::fully_normalize(infcx, traits::ObligationCause::dummy(), cx.param_env, assoc_pred) else {
                    continue;
                };
                // If that predicate doesn't hold modulo regions (but passed during type-check),
                // then we must've taken advantage of the hack in `project_and_unify_types` where
                // we replace opaques with inference vars. Emit a warning!
                if !infcx.predicate_must_hold_modulo_regions(&traits::Obligation::new(
                    cx.tcx,
                    traits::ObligationCause::dummy(),
                    cx.param_env,
                    assoc_pred,
                )) {
                    // If it's a trait bound and an opaque that doesn't satisfy it,
                    // then we can emit a suggestion to add the bound.
                    let add_bound = match (proj_term.kind(), assoc_pred.kind().skip_binder()) {
                        (
                            ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }),
                            ty::ClauseKind::Trait(trait_pred),
                        ) => Some(AddBound {
                            suggest_span: cx.tcx.def_span(*def_id).shrink_to_hi(),
                            trait_ref: trait_pred.print_modifiers_and_trait_path(),
                        }),
                        _ => None,
                    };
                    cx.emit_spanned_lint(
                        OPAQUE_HIDDEN_INFERRED_BOUND,
                        pred_span,
                        OpaqueHiddenInferredBoundLint {
                            ty: cx.tcx.mk_opaque(
                                def_id,
                                ty::InternalSubsts::identity_for_item(cx.tcx, def_id),
                            ),
                            proj_ty: proj_term,
                            assoc_pred_span,
                            add_bound,
                        },
                    );
                }
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_opaque_hidden_inferred_bound)]
struct OpaqueHiddenInferredBoundLint<'tcx> {
    ty: Ty<'tcx>,
    proj_ty: Ty<'tcx>,
    #[label(lint_specifically)]
    assoc_pred_span: Span,
    #[subdiagnostic]
    add_bound: Option<AddBound<'tcx>>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    lint_opaque_hidden_inferred_bound_sugg,
    style = "verbose",
    applicability = "machine-applicable",
    code = " + {trait_ref}"
)]
struct AddBound<'tcx> {
    #[primary_span]
    suggest_span: Span,
    #[skip_arg]
    trait_ref: TraitPredPrintModifiersAndPath<'tcx>,
}

use rustc_hir::{self as hir, AmbigArg};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::print::{PrintTraitPredicateExt as _, TraitPredPrintModifiersAndPath};
use rustc_middle::ty::{self, BottomUpFolder, Ty, TypeFoldable};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{Span, kw};
use rustc_trait_selection::traits::{self, ObligationCtxt};

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
    /// We plan on reintroducing this as a hard error, but in the meantime,
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
    /// impl<F: Duh> Trait for F {
    ///     type Assoc = F;
    /// }
    ///
    /// type Tait = impl Sized;
    ///
    /// #[define_opaque(Tait)]
    /// fn test() -> impl Trait<Assoc = Tait> {
    ///     42
    /// }
    ///
    /// fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// In this example, `test` declares that the associated type `Assoc` for
    /// `impl Trait` is `impl Sized`, which does not satisfy the bound `Duh`
    /// on the associated type.
    ///
    /// Although the hidden type, `i32` does satisfy this bound, we do not
    /// consider the return type to be well-formed with this lint. It can be
    /// fixed by changing `Tait = impl Sized` into `Tait = impl Sized + Duh`.
    pub OPAQUE_HIDDEN_INFERRED_BOUND,
    Warn,
    "detects the use of nested `impl Trait` types in associated type bounds that are not general enough"
}

declare_lint_pass!(OpaqueHiddenInferredBound => [OPAQUE_HIDDEN_INFERRED_BOUND]);

impl<'tcx> LateLintPass<'tcx> for OpaqueHiddenInferredBound {
    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'tcx, AmbigArg>) {
        let hir::TyKind::OpaqueDef(opaque) = &ty.kind else {
            return;
        };

        // If this is an RPITIT from a trait method with no body, then skip.
        // That's because although we may have an opaque type on the function,
        // it won't have a hidden type, so proving predicates about it is
        // not really meaningful.
        if let hir::OpaqueTyOrigin::FnReturn { parent: method_def_id, .. } = opaque.origin
            && let hir::Node::TraitItem(trait_item) = cx.tcx.hir_node_by_def_id(method_def_id)
            && !trait_item.defaultness.has_value()
        {
            return;
        }

        let def_id = opaque.def_id.to_def_id();
        let infcx = &cx.tcx.infer_ctxt().build(cx.typing_mode());
        // For every projection predicate in the opaque type's explicit bounds,
        // check that the type that we're assigning actually satisfies the bounds
        // of the associated type.
        for (pred, pred_span) in cx.tcx.explicit_item_bounds(def_id).iter_identity_copied() {
            infcx.enter_forall(pred.kind(), |predicate| {
                let ty::ClauseKind::Projection(proj) = predicate else {
                    return;
                };
                // Only check types, since those are the only things that may
                // have opaques in them anyways.
                let Some(proj_term) = proj.term.as_type() else { return };

                // HACK: `impl Trait<Assoc = impl Trait2>` from an RPIT is "ok"...
                if let ty::Alias(ty::Opaque, opaque_ty) = *proj_term.kind()
                    && cx.tcx.parent(opaque_ty.def_id) == def_id
                    && matches!(
                        opaque.origin,
                        hir::OpaqueTyOrigin::FnReturn { .. } | hir::OpaqueTyOrigin::AsyncFn { .. }
                    )
                {
                    return;
                }

                // HACK: `async fn() -> Self` in traits is "ok"...
                // This is not really that great, but it's similar to why the `-> Self`
                // return type is well-formed in traits even when `Self` isn't sized.
                if let ty::Param(param_ty) = *proj_term.kind()
                    && param_ty.name == kw::SelfUpper
                    && matches!(
                        opaque.origin,
                        hir::OpaqueTyOrigin::AsyncFn {
                            in_trait_or_impl: Some(hir::RpitContext::Trait),
                            ..
                        }
                    )
                {
                    return;
                }

                let proj_ty = Ty::new_projection_from_args(
                    cx.tcx,
                    proj.projection_term.def_id,
                    proj.projection_term.args,
                );
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
                    .explicit_item_bounds(proj.projection_term.def_id)
                    .iter_instantiated_copied(cx.tcx, proj.projection_term.args)
                {
                    let assoc_pred = assoc_pred.fold_with(proj_replacer);

                    let ocx = ObligationCtxt::new(infcx);
                    let assoc_pred =
                        ocx.normalize(&traits::ObligationCause::dummy(), cx.param_env, assoc_pred);
                    if !ocx.select_all_or_error().is_empty() {
                        // Can't normalize for some reason...?
                        continue;
                    }

                    ocx.register_obligation(traits::Obligation::new(
                        cx.tcx,
                        traits::ObligationCause::dummy(),
                        cx.param_env,
                        assoc_pred,
                    ));

                    // If that predicate doesn't hold modulo regions (but passed during type-check),
                    // then we must've taken advantage of the hack in `project_and_unify_types` where
                    // we replace opaques with inference vars. Emit a warning!
                    if !ocx.select_all_or_error().is_empty() {
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

                        cx.emit_span_lint(
                            OPAQUE_HIDDEN_INFERRED_BOUND,
                            pred_span,
                            OpaqueHiddenInferredBoundLint {
                                ty: Ty::new_opaque(
                                    cx.tcx,
                                    def_id,
                                    ty::GenericArgs::identity_for_item(cx.tcx, def_id),
                                ),
                                proj_ty: proj_term,
                                assoc_pred_span,
                                add_bound,
                            },
                        );
                    }
                }
            });
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

use crate::infer::InferCtxt;
use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::{ObligationCause, PredicateObligation};
use rustc_infer::traits::TraitEngine;
use rustc_middle::ty::{self, ToPredicate, TypeFoldable};

pub(crate) fn update<'tcx, T>(
    engine: &mut T,
    infcx: &InferCtxt<'_, 'tcx>,
    obligation: &PredicateObligation<'tcx>,
) where
    T: TraitEngine<'tcx>,
{
    if let ty::PredicateKind::Trait(predicate) = obligation.predicate.kind().skip_binder() {
        if predicate.trait_ref.def_id
            != infcx.tcx.require_lang_item(rustc_hir::LangItem::Sized, None)
        {
            // fixme: copy of mk_trait_obligation_with_new_self_ty
            let new_self_ty = infcx.tcx.types.unit;

            let trait_ref = ty::TraitRef {
                substs: infcx.tcx.mk_substs_trait(new_self_ty, &predicate.trait_ref.substs[1..]),
                ..predicate.trait_ref
            };

            // Then contstruct a new obligation with Self = () added
            // to the ParamEnv, and see if it holds.
            let o = rustc_infer::traits::Obligation::new(
                ObligationCause::dummy(),
                obligation.param_env,
                obligation
                    .predicate
                    .kind()
                    .map_bound(|_| {
                        ty::PredicateKind::Trait(ty::TraitPredicate {
                            trait_ref,
                            constness: predicate.constness,
                        })
                    })
                    .to_predicate(infcx.tcx),
            );
            // Don't report overflow errors. Otherwise equivalent to may_hold.
            if let Ok(result) = infcx.probe(|_| infcx.evaluate_obligation(&o)) {
                if result.may_apply() {
                    if let Some(ty) = infcx
                        .shallow_resolve(predicate.self_ty())
                        .ty_vid()
                        .map(|t| infcx.root_var(t))
                    {
                        debug!("relationship: {:?}.self_in_trait = true", ty);
                        engine.relationships().entry(ty).or_default().self_in_trait = true;
                    } else {
                        debug!("relationship: did not find TyVid for self ty...");
                    }
                }
            }
        }
    }

    if let ty::PredicateKind::Projection(predicate) = obligation.predicate.kind().skip_binder() {
        // If the projection predicate (Foo::Bar == X) has X as a non-TyVid,
        // we need to make it into one.
        if let Some(vid) = predicate.ty.ty_vid() {
            debug!("relationship: {:?}.output = true", vid);
            engine.relationships().entry(vid).or_default().output = true;
        } else {
            // This will have registered a projection obligation that will hit
            // the Some(vid) branch above. So we don't need to do anything
            // further here.
            debug!(
                "skipping relationship for obligation {:?} -- would need to normalize",
                obligation
            );
            if !predicate.projection_ty.has_escaping_bound_vars() {
                // FIXME: We really *should* do this even with escaping bound
                // vars, but there's not much we can do here. In the worst case
                // (if this ends up being important) we just don't register a relationship and then end up falling back to !,
                // which is not terrible.

                //engine.normalize_projection_type(
                //    infcx,
                //    obligation.param_env,
                //    predicate.projection_ty,
                //    obligation.cause.clone(),
                //);
            }
        }
    }
}

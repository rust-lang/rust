use crate::infer::InferCtxt;
use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::{ObligationCause, PredicateObligation};
use rustc_infer::traits::TraitEngine;
use rustc_middle::ty::{self, ToPredicate};

pub(crate) fn update<'tcx, T>(
    engine: &mut T,
    infcx: &InferCtxt<'_, 'tcx>,
    obligation: &PredicateObligation<'tcx>,
) where
    T: TraitEngine<'tcx>,
{
    // (*) binder skipped
    if let ty::PredicateKind::Trait(predicate) = obligation.predicate.kind().skip_binder() {
        if let Some(ty) =
            infcx.shallow_resolve(predicate.self_ty()).ty_vid().map(|t| infcx.root_var(t))
        {
            if infcx
                .tcx
                .lang_items()
                .sized_trait()
                .map_or(false, |st| st != predicate.trait_ref.def_id)
            {
                let new_self_ty = infcx.tcx.types.unit;

                let trait_ref = ty::TraitRef {
                    substs: infcx
                        .tcx
                        .mk_substs_trait(new_self_ty, &predicate.trait_ref.substs[1..]),
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
                            // (*) binder moved here
                            ty::PredicateKind::Trait(ty::TraitPredicate {
                                trait_ref,
                                constness: predicate.constness,
                                polarity: predicate.polarity,
                            })
                        })
                        .to_predicate(infcx.tcx),
                );
                // Don't report overflow errors. Otherwise equivalent to may_hold.
                if let Ok(result) = infcx.probe(|_| infcx.evaluate_obligation(&o)) {
                    if result.may_apply() {
                        engine.relationships().entry(ty).or_default().self_in_trait = true;
                    }
                }
            }
        }
    }

    if let ty::PredicateKind::Projection(predicate) = obligation.predicate.kind().skip_binder() {
        // If the projection predicate (Foo::Bar == X) has X as a non-TyVid,
        // we need to make it into one.
        if let Some(vid) = predicate.term.ty().and_then(|ty| ty.ty_vid()) {
            debug!("relationship: {:?}.output = true", vid);
            engine.relationships().entry(vid).or_default().output = true;
        }
    }
}

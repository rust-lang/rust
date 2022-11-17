use crate::infer::InferCtxt;
use crate::traits::query::evaluate_obligation::InferCtxtExt;
use crate::traits::PredicateObligation;
use rustc_infer::traits::TraitEngine;
use rustc_middle::ty;

pub(crate) fn update<'tcx, T>(
    engine: &mut T,
    infcx: &InferCtxt<'tcx>,
    obligation: &PredicateObligation<'tcx>,
) where
    T: TraitEngine<'tcx>,
{
    // (*) binder skipped
    if let ty::PredicateKind::Trait(tpred) = obligation.predicate.kind().skip_binder()
        && let Some(ty) = infcx.shallow_resolve(tpred.self_ty()).ty_vid().map(|t| infcx.root_var(t))
        && infcx.tcx.lang_items().sized_trait().map_or(false, |st| st != tpred.trait_ref.def_id)
    {
        let new_self_ty = infcx.tcx.types.unit;

        let trait_ref = infcx.tcx.mk_trait_ref(
            tpred.trait_ref.def_id,
            new_self_ty, &tpred.trait_ref.substs[1..],
        );

        // Then construct a new obligation with Self = () added
        // to the ParamEnv, and see if it holds.
        let o = obligation.with(infcx.tcx,
            obligation
                .predicate
                .kind()
                .rebind(
                    // (*) binder moved here
                    ty::PredicateKind::Trait(ty::TraitPredicate {
                        trait_ref,
                        constness: tpred.constness,
                        polarity: tpred.polarity,
                    })
                ),
        );
        // Don't report overflow errors. Otherwise equivalent to may_hold.
        if let Ok(result) = infcx.probe(|_| infcx.evaluate_obligation(&o)) && result.may_apply() {
            engine.relationships().entry(ty).or_default().self_in_trait = true;
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

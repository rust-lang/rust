use crate::infer::canonical::{Canonicalized, CanonicalizedQueryResponse};
use crate::traits::query::Fallible;
use rustc_middle::ty::{self, ParamEnvAnd, TyCtxt};

pub use rustc_middle::traits::query::type_op::ProvePredicate;

impl<'tcx> super::QueryTypeOp<'tcx> for ProvePredicate<'tcx> {
    type QueryResponse = ();

    fn try_fast_path(
        tcx: TyCtxt<'tcx>,
        key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        // Proving Sized, very often on "obviously sized" types like
        // `&T`, accounts for about 60% percentage of the predicates
        // we have to prove. No need to canonicalize and all that for
        // such cases.
        if let ty::PredicateKind::Trait(trait_ref) = key.value.predicate.kind().skip_binder() {
            if let Some(sized_def_id) = tcx.lang_items().sized_trait() {
                if trait_ref.def_id() == sized_def_id {
                    if trait_ref.self_ty().is_trivially_sized(tcx) {
                        return Some(());
                    }
                }
            }
        }

        None
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        mut canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, ()>> {
        match canonicalized.value.value.predicate.kind().skip_binder() {
            ty::PredicateKind::Trait(pred) => {
                canonicalized.value.param_env.remap_constness_with(pred.constness);
            }
            _ => canonicalized.value.param_env = canonicalized.value.param_env.without_const(),
        }
        tcx.type_op_prove_predicate(canonicalized)
    }
}

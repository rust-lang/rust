use crate::infer::canonical::{Canonical, Canonicalized, CanonicalizedQueryResponse, QueryResponse};
use crate::traits::query::Fallible;
use crate::ty::{ParamEnvAnd, Predicate, TyCtxt};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ProvePredicate<'tcx> {
    pub predicate: Predicate<'tcx>,
}

impl<'tcx> ProvePredicate<'tcx> {
    pub fn new(predicate: Predicate<'tcx>) -> Self {
        ProvePredicate { predicate }
    }
}

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
        if let Predicate::Trait(trait_ref) = key.value.predicate {
            if let Some(sized_def_id) = tcx.lang_items().sized_trait() {
                if trait_ref.def_id() == sized_def_id {
                    if trait_ref.skip_binder().self_ty().is_trivially_sized(tcx) {
                        return Some(());
                    }
                }
            }
        }

        None
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, ()>> {
        tcx.type_op_prove_predicate(canonicalized)
    }

    fn shrink_to_tcx_lifetime(
        v: &'a CanonicalizedQueryResponse<'tcx, ()>,
    ) -> &'a Canonical<'tcx, QueryResponse<'tcx, ()>> {
        v
    }
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ProvePredicate<'tcx> {
        predicate,
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for ProvePredicate<'a> {
        type Lifted = ProvePredicate<'tcx>;
        predicate,
    }
}

impl_stable_hash_for! {
    struct ProvePredicate<'tcx> { predicate }
}

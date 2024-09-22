use rustc_hir::LangItem;
use rustc_infer::traits::Obligation;
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::query::NoSolution;
pub use rustc_middle::traits::query::type_op::ProvePredicate;
use rustc_middle::ty::{self, ParamEnvAnd, TyCtxt};

use crate::infer::canonical::{Canonical, CanonicalQueryResponse};
use crate::traits::ObligationCtxt;

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
        if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_ref)) =
            key.value.predicate.kind().skip_binder()
            && tcx.is_lang_item(trait_ref.def_id(), LangItem::Sized)
            && trait_ref.self_ty().is_trivially_sized(tcx)
        {
            return Some(());
        }

        if let ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg)) =
            key.value.predicate.kind().skip_binder()
        {
            match arg.as_type()?.kind() {
                ty::Param(_)
                | ty::Bool
                | ty::Char
                | ty::Int(_)
                | ty::Float(_)
                | ty::Str
                | ty::Uint(_) => {
                    return Some(());
                }
                _ => {}
            }
        }

        None
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, ()>, NoSolution> {
        tcx.type_op_prove_predicate(canonicalized)
    }

    fn perform_locally_with_next_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
    ) -> Result<Self::QueryResponse, NoSolution> {
        ocx.register_obligation(Obligation::new(
            ocx.infcx.tcx,
            ObligationCause::dummy(),
            key.param_env,
            key.value.predicate,
        ));
        Ok(())
    }
}

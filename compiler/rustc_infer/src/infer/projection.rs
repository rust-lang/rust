use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{self, Ty};

use super::InferCtxt;
use crate::traits::{Obligation, PredicateObligations};

impl<'tcx> InferCtxt<'tcx> {
    /// Instead of normalizing an associated type projection,
    /// this function generates an inference variable and registers
    /// an obligation that this inference variable must be the result
    /// of the given projection. This allows us to proceed with projections
    /// while they cannot be resolved yet due to missing information or
    /// simply due to the lack of access to the trait resolution machinery.
    pub fn projection_ty_to_infer(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        projection_ty: ty::AliasTy<'tcx>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        obligations: &mut PredicateObligations<'tcx>,
    ) -> Ty<'tcx> {
        debug_assert!(!self.next_trait_solver());
        let ty_var = self.next_ty_var(self.tcx.def_span(projection_ty.def_id));
        let projection =
            ty::PredicateKind::Clause(ty::ClauseKind::Projection(ty::ProjectionPredicate {
                projection_term: projection_ty.into(),
                term: ty_var.into(),
            }));
        let obligation =
            Obligation::with_depth(self.tcx, cause, recursion_depth, param_env, projection);
        obligations.push(obligation);
        ty_var
    }
}

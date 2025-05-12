use rustc_middle::traits::ObligationCause;
use rustc_middle::ty;

use super::InferCtxt;
use crate::infer::Term;
use crate::traits::{Obligation, PredicateObligations};

impl<'tcx> InferCtxt<'tcx> {
    /// Instead of normalizing an associated type projection,
    /// this function generates an inference variable and registers
    /// an obligation that this inference variable must be the result
    /// of the given projection. This allows us to proceed with projections
    /// while they cannot be resolved yet due to missing information or
    /// simply due to the lack of access to the trait resolution machinery.
    pub fn projection_term_to_infer(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        alias_term: ty::AliasTerm<'tcx>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        obligations: &mut PredicateObligations<'tcx>,
    ) -> Term<'tcx> {
        debug_assert!(!self.next_trait_solver());

        let span = self.tcx.def_span(alias_term.def_id);
        let infer_var = if alias_term.kind(self.tcx).is_type() {
            self.next_ty_var(span).into()
        } else {
            self.next_const_var(span).into()
        };

        let projection =
            ty::PredicateKind::Clause(ty::ClauseKind::Projection(ty::ProjectionPredicate {
                projection_term: alias_term,
                term: infer_var,
            }));
        let obligation =
            Obligation::with_depth(self.tcx, cause, recursion_depth, param_env, projection);
        obligations.push(obligation);

        infer_var
    }
}

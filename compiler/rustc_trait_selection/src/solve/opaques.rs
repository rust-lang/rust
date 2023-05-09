use rustc_infer::infer::InferOk;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::traits::{ObligationCause, Reveal};
use rustc_middle::ty::ProjectionPredicate;

use super::{EvalCtxt, SolverMode};

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn normalize_opaque_type(
        &mut self,
        goal: Goal<'tcx, ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let opaque_ty = goal.predicate.projection_ty;
        let expected = goal.predicate.term.ty().expect("no such thing as an opaque const");

        match goal.param_env.reveal() {
            Reveal::UserFacing => match self.solver_mode() {
                SolverMode::Normal => self.probe(|ecx| {
                    // FIXME: Check that the usage is "defining" (all free params), otherwise bail.
                    // FIXME: This should probably just check the anchor directly
                    let InferOk { value: (), obligations } = self.infcx.handle_opaque_type(
                        expected,
                        tcx.mk_opaque(opaque_ty.def_id, opaque_ty.substs),
                        true,
                        &ObligationCause::dummy(),
                        goal.param_env,
                    )?;
                    // FIXME: Need to fold these to replace the opaque ty with the expected ty.
                    ecx.add_goals(obligations.into_iter().map(Into::into));
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }),
                SolverMode::Coherence => {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                }
            },
            Reveal::All => self.probe(|ecx| {
                let actual = tcx.type_of(opaque_ty.def_id).subst(tcx, opaque_ty.substs);
                ecx.eq(goal.param_env, expected, actual)?;
                ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }),
        }
    }
}

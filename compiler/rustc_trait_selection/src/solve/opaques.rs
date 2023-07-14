use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::traits::Reveal;
use rustc_middle::ty;
use rustc_middle::ty::util::NotUniqueParam;

use super::{EvalCtxt, SolverMode};

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn normalize_opaque_type(
        &mut self,
        goal: Goal<'tcx, ty::ProjectionPredicate<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let opaque_ty = goal.predicate.projection_ty;
        let expected = goal.predicate.term.ty().expect("no such thing as an opaque const");

        match (goal.param_env.reveal(), self.solver_mode()) {
            (Reveal::UserFacing, SolverMode::Normal) => {
                let Some(opaque_ty_def_id) = opaque_ty.def_id.as_local() else {
                    return Err(NoSolution);
                };
                // FIXME: at some point we should call queries without defining
                // new opaque types but having the existing opaque type definitions.
                // This will require moving this below "Prefer opaques registered already".
                if !self.can_define_opaque_ty(opaque_ty_def_id) {
                    return Err(NoSolution);
                }
                // FIXME: This may have issues when the args contain aliases...
                match self.tcx().uses_unique_placeholders_ignoring_regions(opaque_ty.args) {
                    Err(NotUniqueParam::NotParam(param)) if param.is_non_region_infer() => {
                        return self.evaluate_added_goals_and_make_canonical_response(
                            Certainty::AMBIGUOUS,
                        );
                    }
                    Err(_) => {
                        return Err(NoSolution);
                    }
                    Ok(()) => {}
                }
                // Prefer opaques registered already.
                let opaque_type_key =
                    ty::OpaqueTypeKey { def_id: opaque_ty_def_id, args: opaque_ty.args };
                let matches =
                    self.unify_existing_opaque_tys(goal.param_env, opaque_type_key, expected);
                if !matches.is_empty() {
                    if let Some(response) = self.try_merge_responses(&matches) {
                        return Ok(response);
                    } else {
                        return self.flounder(&matches);
                    }
                }
                // Otherwise, define a new opaque type
                self.insert_hidden_type(opaque_type_key, goal.param_env, expected)?;
                self.add_item_bounds_for_hidden_type(
                    opaque_ty.def_id,
                    opaque_ty.args,
                    goal.param_env,
                    expected,
                );
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            (Reveal::UserFacing, SolverMode::Coherence) => {
                // An impossible opaque type bound is the only way this goal will fail
                // e.g. assigning `impl Copy := NotCopy`
                self.add_item_bounds_for_hidden_type(
                    opaque_ty.def_id,
                    opaque_ty.args,
                    goal.param_env,
                    expected,
                );
                self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
            }
            (Reveal::All, _) => {
                // FIXME: Add an assertion that opaque type storage is empty.
                let actual = tcx.type_of(opaque_ty.def_id).instantiate(tcx, opaque_ty.args);
                self.eq(goal.param_env, expected, actual)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
        }
    }
}

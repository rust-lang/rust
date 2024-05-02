//! Computes a normalizes-to (projection) goal for opaque types. This goal
//! behaves differently depending on the param-env's reveal mode and whether
//! the opaque is in a defining scope.
use rustc_infer::infer::InferCtxt;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{Certainty, Goal, QueryResult};
use rustc_middle::traits::TreatOpaque;
use rustc_middle::ty;
use rustc_middle::ty::util::NotUniqueParam;

use crate::solve::EvalCtxt;

impl<'tcx> EvalCtxt<'_, InferCtxt<'tcx>> {
    pub(super) fn normalize_opaque_type(
        &mut self,
        goal: Goal<'tcx, ty::NormalizesTo<'tcx>>,
    ) -> QueryResult<'tcx> {
        let tcx = self.tcx();
        let opaque_ty = goal.predicate.alias;
        let expected = goal.predicate.term.ty().expect("no such thing as an opaque const");

        match self.treat_opaque_ty(goal.param_env.reveal(), opaque_ty.def_id) {
            TreatOpaque::Rigid => Err(NoSolution),
            TreatOpaque::Ambiguous => {
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
            TreatOpaque::Reveal => {
                // FIXME: Add an assertion that opaque type storage is empty.
                let actual = tcx.type_of(opaque_ty.def_id).instantiate(tcx, opaque_ty.args);
                let actual = tcx.fold_regions(actual, |re, _| match *re {
                    ty::ReErased => self.next_region_infer(),
                    _ => re,
                });
                self.eq(goal.param_env, expected, actual)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            TreatOpaque::Define => {
                // FIXME: at some point we should call queries without defining
                // new opaque types but having the existing opaque type definitions.
                // This will require moving this below "Prefer opaques registered already".
                let Some(opaque_ty_def_id) = opaque_ty.def_id.as_local() else {
                    return Err(NoSolution);
                };
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
                // FIXME: This also unifies the previous hidden type with the expected.
                //
                // If that fails, we insert `expected` as a new hidden type instead of
                // eagerly emitting an error.
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
        }
    }
}

//! Computes a normalizes-to (projection) goal for opaque types. This goal
//! behaves differently depending on the current `TypingMode`.

use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::GoalSource;
use rustc_type_ir::{self as ty, Interner, TypingMode, fold_regions};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, QueryResult};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(super) fn normalize_opaque_type(
        &mut self,
        goal: Goal<I, ty::NormalizesTo<I>>,
    ) -> QueryResult<I> {
        let cx = self.cx();
        let opaque_ty = goal.predicate.alias;
        let expected = goal.predicate.term.as_type().expect("no such thing as an opaque const");

        match self.typing_mode() {
            TypingMode::Coherence => {
                // An impossible opaque type bound is the only way this goal will fail
                // e.g. assigning `impl Copy := NotCopy`
                self.add_item_bounds_for_hidden_type(
                    opaque_ty.def_id,
                    opaque_ty.args,
                    goal.param_env,
                    expected,
                );
                // Trying to normalize an opaque type during coherence is always ambiguous.
                // We add a nested ambiguous goal here instead of using `Certainty::AMBIGUOUS`.
                // This allows us to return the nested goals to the parent `AliasRelate` goal.
                // This can then allow nested goals to fail after we've constrained the `term`.
                self.add_goal(GoalSource::Misc, goal.with(cx, ty::PredicateKind::Ambiguous));
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            TypingMode::Analysis {
                defining_opaque_types_and_generators: defining_opaque_types,
            }
            | TypingMode::Borrowck { defining_opaque_types } => {
                let Some(def_id) = opaque_ty
                    .def_id
                    .as_local()
                    .filter(|&def_id| defining_opaque_types.contains(&def_id))
                else {
                    // If we're not in the defining scope, treat the alias as rigid.
                    self.structurally_instantiate_normalizes_to_term(goal, goal.predicate.alias);
                    return self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
                };

                // We structurally normalize the args so that we're able to detect defining uses
                // later on.
                //
                // This reduces the amount of duplicate definitions in the `opaque_type_storage` and
                // strengthens inference. This causes us to subtly depend on the normalization behavior
                // when inferring the hidden type of opaques.
                //
                // E.g. it's observable that we don't normalize nested aliases with bound vars in
                // `structurally_normalize` and because we use structural lookup, we also don't
                // reuse an entry for `Tait<for<'a> fn(&'a ())>` for `Tait<for<'b> fn(&'b ())>`.
                let normalized_args =
                    cx.mk_args_from_iter(opaque_ty.args.iter().map(|arg| match arg.kind() {
                        ty::GenericArgKind::Lifetime(lt) => Ok(lt.into()),
                        ty::GenericArgKind::Type(ty) => {
                            self.structurally_normalize_ty(goal.param_env, ty).map(Into::into)
                        }
                        ty::GenericArgKind::Const(ct) => {
                            self.structurally_normalize_const(goal.param_env, ct).map(Into::into)
                        }
                    }))?;

                let opaque_type_key = ty::OpaqueTypeKey { def_id, args: normalized_args };
                if let Some(prev) = self.register_hidden_type_in_storage(opaque_type_key, expected)
                {
                    self.eq(goal.param_env, expected, prev)?;
                } else {
                    // During HIR typeck, opaque types start out as unconstrained
                    // inference variables. In borrowck we instead use the type
                    // computed in HIR typeck as the initial value.
                    match self.typing_mode() {
                        TypingMode::Analysis { .. } => {}
                        TypingMode::Borrowck { .. } => {
                            let actual = cx
                                .type_of_opaque_hir_typeck(def_id)
                                .instantiate(cx, opaque_ty.args);
                            let actual = fold_regions(cx, actual, |re, _dbi| match re.kind() {
                                ty::ReErased => self.next_region_var(),
                                _ => re,
                            });
                            self.eq(goal.param_env, expected, actual)?;
                        }
                        _ => unreachable!(),
                    }
                }

                self.add_item_bounds_for_hidden_type(
                    def_id.into(),
                    normalized_args,
                    goal.param_env,
                    expected,
                );
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            TypingMode::PostBorrowckAnalysis { defined_opaque_types } => {
                let Some(def_id) = opaque_ty
                    .def_id
                    .as_local()
                    .filter(|&def_id| defined_opaque_types.contains(&def_id))
                else {
                    self.structurally_instantiate_normalizes_to_term(goal, goal.predicate.alias);
                    return self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes);
                };

                let actual = cx.type_of(def_id.into()).instantiate(cx, opaque_ty.args);
                // FIXME: Actually use a proper binder here instead of relying on `ReErased`.
                //
                // This is also probably unsound or sth :shrug:
                let actual = fold_regions(cx, actual, |re, _dbi| match re.kind() {
                    ty::ReErased => self.next_region_var(),
                    _ => re,
                });
                self.eq(goal.param_env, expected, actual)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            TypingMode::PostAnalysis => {
                // FIXME: Add an assertion that opaque type storage is empty.
                let actual = cx.type_of(opaque_ty.def_id).instantiate(cx, opaque_ty.args);
                self.eq(goal.param_env, expected, actual)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
        }
    }
}

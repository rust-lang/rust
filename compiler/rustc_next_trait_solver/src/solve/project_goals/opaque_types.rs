//! Computes a projection goal for opaque types. This goal
//! behaves differently depending on the current `TypingMode`.

use rustc_type_ir::inherent::*;
use rustc_type_ir::solve::{GoalSource, QueryResultOrRerunNonErased, RerunReason};
use rustc_type_ir::{self as ty, Interner, MayBeErased, TypingMode, fold_regions};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal};

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    #[tracing::instrument(skip(self))]
    pub(super) fn normalize_opaque_type(
        &mut self,
        goal: Goal<I, ty::ProjectionPredicate<I>>,
    ) -> QueryResultOrRerunNonErased<I> {
        let cx = self.cx();
        let opaque_ty = goal.predicate.projection_term;
        let expected = goal.predicate.term.as_type().expect("no such thing as an opaque const");
        let def_id = opaque_ty.expect_opaque_ty_def_id();

        match self.typing_mode() {
            TypingMode::Coherence => {
                // An impossible opaque type bound is the only way this goal will fail
                // e.g. assigning `impl Copy := NotCopy`
                self.add_item_bounds_for_hidden_type(
                    def_id,
                    opaque_ty.args,
                    goal.param_env,
                    expected,
                )?;
                // Trying to normalize an opaque type during coherence is always ambiguous.
                // We add a nested ambiguous goal here instead of using `Certainty::AMBIGUOUS`.
                // This allows us to return the nested goals to the parent `AliasRelate` goal.
                // This can then allow nested goals to fail after we've constrained the `term`.
                self.add_goal(GoalSource::Misc, goal.with(cx, ty::PredicateKind::Ambiguous))?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    .map_err(Into::into)
            }
            TypingMode::Typeck { defining_opaque_types_and_generators: defining_opaque_types }
            | TypingMode::PostTypeckUntilBorrowck { defining_opaque_types } => {
                let Some(def_id) = def_id
                    .as_local()
                    .filter(|&def_id| defining_opaque_types.contains(&def_id.into()))
                else {
                    // If we're not in the defining scope, treat the alias as rigid.
                    self.eq(
                        goal.param_env,
                        opaque_ty.to_term(cx, ty::IsRigid::Yes),
                        goal.predicate.term,
                    )?;
                    return self
                        .evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                        .map_err(Into::into);
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
                    match self.typing_mode().assert_not_erased() {
                        TypingMode::Typeck { .. } => {}
                        TypingMode::PostTypeckUntilBorrowck { .. } => {
                            let actual = cx
                                .type_of_opaque_hir_typeck(def_id)
                                .instantiate(cx, opaque_ty.args);
                            let actual = actual.map(|v| {
                                fold_regions(cx, v, |re, _dbi| match re.kind() {
                                    ty::ReErased => self.next_region_var(),
                                    _ => re,
                                })
                            });
                            let actual =
                                self.normalize(GoalSource::Misc, goal.param_env, actual)?;
                            self.eq(goal.param_env, expected, actual)?;
                        }
                        TypingMode::Coherence
                        | TypingMode::PostBorrowck { .. }
                        | TypingMode::PostAnalysis
                        | TypingMode::Reflection
                        | TypingMode::Codegen => unreachable!(),
                    }
                }

                self.add_item_bounds_for_hidden_type(
                    def_id.into(),
                    normalized_args,
                    goal.param_env,
                    expected,
                )?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    .map_err(Into::into)
            }
            TypingMode::PostBorrowck { defined_opaque_types } => {
                let Some(def_id) = def_id
                    .as_local()
                    .filter(|&def_id| defined_opaque_types.contains(&def_id.into()))
                else {
                    self.eq(
                        goal.param_env,
                        opaque_ty.to_term(cx, ty::IsRigid::Yes),
                        goal.predicate.term,
                    )?;
                    return self
                        .evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                        .map_err(Into::into);
                };

                let actual = cx.type_of(def_id.into()).instantiate(cx, opaque_ty.args);
                // FIXME: Actually use a proper binder here instead of relying on `ReErased`.
                //
                // This is also probably unsound or sth :shrug:
                let actual = actual.map(|v| {
                    fold_regions(cx, v, |re, _dbi| match re.kind() {
                        ty::ReErased => self.next_region_var(),
                        _ => re,
                    })
                });
                let actual = self.normalize(GoalSource::Misc, goal.param_env, actual)?;
                self.eq(goal.param_env, expected, actual)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    .map_err(Into::into)
            }
            // FIXME(try_as_dyn): probably want to treat opaques opaquely and rigid
            TypingMode::Reflection | TypingMode::PostAnalysis | TypingMode::Codegen => {
                // FIXME: Add an assertion that opaque type storage is empty.
                let actual = cx.type_of(def_id.into()).instantiate(cx, opaque_ty.args);
                let actual = self.normalize(GoalSource::Misc, goal.param_env, actual)?;
                self.eq(goal.param_env, expected, actual)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    .map_err(Into::into)
            }
            TypingMode::ErasedNotCoherence(MayBeErased) => {
                // If we have a local defid, in other typing modes we check whether
                // this is the defining scope, and otherwise treat it as rigid.
                // However, in `ErasedNotcoherence` we *always* treat it as rigid.
                // This is the same as other modes if def_id is None, but wrong if we do have a DefId.
                // So, if we have one, we register in the EvalCtxt that we may need that defid.
                // We might then decide to rerun in the correct typing mode.
                if let Some(def_id) = def_id.as_local() {
                    self.opaque_accesses.rerun_if_opaque_in_opaque_type_storage(
                        RerunReason::NormalizeOpaqueType,
                        def_id,
                    )?;
                } else {
                    self.opaque_accesses
                        .rerun_if_in_post_analysis(RerunReason::NormalizeOpaqueTypeRemoteCrate)?;
                }

                // Always treat the opaque type as rigid.
                self.eq(
                    goal.param_env,
                    opaque_ty.to_term(cx, ty::IsRigid::Yes),
                    goal.predicate.term,
                )?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                    .map_err(Into::into)
            }
        }
    }
}

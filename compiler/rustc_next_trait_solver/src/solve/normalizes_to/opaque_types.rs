//! Computes a normalizes-to (projection) goal for opaque types. This goal
//! behaves differently depending on the param-env's reveal mode and whether
//! the opaque is in a defining scope.

use rustc_index::bit_set::GrowableBitSet;
use rustc_type_ir::inherent::*;
use rustc_type_ir::{self as ty, Interner};

use crate::delegate::SolverDelegate;
use crate::solve::{
    Certainty, EvalCtxt, Goal, NoSolution, QueryResult, Reveal, SolverMode, inspect,
};

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
                match uses_unique_placeholders_ignoring_regions(self.cx(), opaque_ty.args) {
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
                let existing = self.probe_existing_opaque_ty(opaque_type_key);
                if let Some((candidate_key, candidate_ty)) = existing {
                    return self
                        .probe(|result| inspect::ProbeKind::OpaqueTypeStorageLookup {
                            result: *result,
                        })
                        .enter(|ecx| {
                            for (a, b) in std::iter::zip(
                                candidate_key.args.iter(),
                                opaque_type_key.args.iter(),
                            ) {
                                ecx.eq(goal.param_env, a, b)?;
                            }
                            ecx.eq(goal.param_env, candidate_ty, expected)?;
                            ecx.add_item_bounds_for_hidden_type(
                                candidate_key.def_id.into(),
                                candidate_key.args,
                                goal.param_env,
                                candidate_ty,
                            );
                            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                        });
                }

                // Otherwise, define a new opaque type
                // FIXME: should we use `inject_hidden_type_unchecked` here?
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
                let actual = cx.type_of(opaque_ty.def_id).instantiate(cx, opaque_ty.args);
                self.eq(goal.param_env, expected, actual)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
        }
    }
}

/// Checks whether each generic argument is simply a unique generic placeholder.
///
/// FIXME: Interner argument is needed to constrain the `I` parameter.
fn uses_unique_placeholders_ignoring_regions<I: Interner>(
    _cx: I,
    args: I::GenericArgs,
) -> Result<(), NotUniqueParam<I>> {
    let mut seen = GrowableBitSet::default();
    for arg in args.iter() {
        match arg.kind() {
            // Ignore regions, since we can't resolve those in a canonicalized
            // query in the trait solver.
            ty::GenericArgKind::Lifetime(_) => {}
            ty::GenericArgKind::Type(t) => match t.kind() {
                ty::Placeholder(p) => {
                    if !seen.insert(p.var()) {
                        return Err(NotUniqueParam::DuplicateParam(t.into()));
                    }
                }
                _ => return Err(NotUniqueParam::NotParam(t.into())),
            },
            ty::GenericArgKind::Const(c) => match c.kind() {
                ty::ConstKind::Placeholder(p) => {
                    if !seen.insert(p.var()) {
                        return Err(NotUniqueParam::DuplicateParam(c.into()));
                    }
                }
                _ => return Err(NotUniqueParam::NotParam(c.into())),
            },
        }
    }

    Ok(())
}

// FIXME: This should check for dupes and non-params first, then infer vars.
enum NotUniqueParam<I: Interner> {
    DuplicateParam(I::GenericArg),
    NotParam(I::GenericArg),
}

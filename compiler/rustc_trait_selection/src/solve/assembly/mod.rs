//! Code shared by trait and projection goals for candidate assembly.

use super::{EvalCtxt, SolverMode};
use crate::traits::coherence;
use rustc_hir::def_id::DefId;
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::Reveal;
use rustc_middle::traits::solve::inspect::ProbeKind;
use rustc_middle::traits::solve::{
    CandidateSource, CanonicalResponse, Certainty, Goal, QueryResult,
};
use rustc_middle::traits::BuiltinImplSource;
use rustc_middle::ty::fast_reject::{SimplifiedType, TreatParams};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{fast_reject, TypeFoldable};
use rustc_middle::ty::{ToPredicate, TypeVisitableExt};
use rustc_span::ErrorGuaranteed;
use std::fmt::Debug;

pub(super) mod structural_traits;

/// A candidate is a possible way to prove a goal.
///
/// It consists of both the `source`, which describes how that goal would be proven,
/// and the `result` when using the given `source`.
#[derive(Debug, Clone)]
pub(super) struct Candidate<'tcx> {
    pub(super) source: CandidateSource,
    pub(super) result: CanonicalResponse<'tcx>,
}

/// Methods used to assemble candidates for either trait or projection goals.
pub(super) trait GoalKind<'tcx>:
    TypeFoldable<TyCtxt<'tcx>> + Copy + Eq + std::fmt::Display
{
    fn self_ty(self) -> Ty<'tcx>;

    fn trait_ref(self, tcx: TyCtxt<'tcx>) -> ty::TraitRef<'tcx>;

    fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self;

    fn trait_def_id(self, tcx: TyCtxt<'tcx>) -> DefId;

    /// Try equating an assumption predicate against a goal's predicate. If it
    /// holds, then execute the `then` callback, which should do any additional
    /// work, then produce a response (typically by executing
    /// [`EvalCtxt::evaluate_added_goals_and_make_canonical_response`]).
    fn probe_and_match_goal_against_assumption(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Clause<'tcx>,
        then: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> QueryResult<'tcx>,
    ) -> QueryResult<'tcx>;

    /// Consider a clause, which consists of a "assumption" and some "requirements",
    /// to satisfy a goal. If the requirements hold, then attempt to satisfy our
    /// goal by equating it with the assumption.
    fn consider_implied_clause(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Clause<'tcx>,
        requirements: impl IntoIterator<Item = Goal<'tcx, ty::Predicate<'tcx>>>,
    ) -> QueryResult<'tcx> {
        Self::probe_and_match_goal_against_assumption(ecx, goal, assumption, |ecx| {
            ecx.add_goals(requirements);
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    /// Consider a bound originating from the item bounds of an alias. For this we
    /// require that the well-formed requirements of the self type of the goal
    /// are "satisfied from the param-env".
    /// See [`EvalCtxt::validate_alias_bound_self_from_param_env`].
    fn consider_alias_bound_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Clause<'tcx>,
    ) -> QueryResult<'tcx> {
        Self::probe_and_match_goal_against_assumption(ecx, goal, assumption, |ecx| {
            ecx.validate_alias_bound_self_from_param_env(goal)
        })
    }

    /// Consider a clause specifically for a `dyn Trait` self type. This requires
    /// additionally checking all of the supertraits and object bounds to hold,
    /// since they're not implied by the well-formedness of the object type.
    fn consider_object_bound_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        assumption: ty::Clause<'tcx>,
    ) -> QueryResult<'tcx> {
        Self::probe_and_match_goal_against_assumption(ecx, goal, assumption, |ecx| {
            let tcx = ecx.tcx();
            let ty::Dynamic(bounds, _, _) = *goal.predicate.self_ty().kind() else {
                bug!("expected object type in `consider_object_bound_candidate`");
            };
            ecx.add_goals(structural_traits::predicates_for_object_candidate(
                &ecx,
                goal.param_env,
                goal.predicate.trait_ref(tcx),
                bounds,
            ));
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_impl_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        impl_def_id: DefId,
    ) -> QueryResult<'tcx>;

    /// If the predicate contained an error, we want to avoid emitting unnecessary trait
    /// errors but still want to emit errors for other trait goals. We have some special
    /// handling for this case.
    ///
    /// Trait goals always hold while projection goals never do. This is a bit arbitrary
    /// but prevents incorrect normalization while hiding any trait errors.
    fn consider_error_guaranteed_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        guar: ErrorGuaranteed,
    ) -> QueryResult<'tcx>;

    /// A type implements an `auto trait` if its components do as well.
    ///
    /// These components are given by built-in rules from
    /// [`structural_traits::instantiate_constituent_tys_for_auto_trait`].
    fn consider_auto_trait_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// A trait alias holds if the RHS traits and `where` clauses hold.
    fn consider_trait_alias_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// A type is `Copy` or `Clone` if its components are `Sized`.
    ///
    /// These components are given by built-in rules from
    /// [`structural_traits::instantiate_constituent_tys_for_sized_trait`].
    fn consider_builtin_sized_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// A type is `Copy` or `Clone` if its components are `Copy` or `Clone`.
    ///
    /// These components are given by built-in rules from
    /// [`structural_traits::instantiate_constituent_tys_for_copy_clone_trait`].
    fn consider_builtin_copy_clone_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// A type is `PointerLike` if we can compute its layout, and that layout
    /// matches the layout of `usize`.
    fn consider_builtin_pointer_like_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// A type is a `FnPtr` if it is of `FnPtr` type.
    fn consider_builtin_fn_ptr_trait_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// A callable type (a closure, fn def, or fn ptr) is known to implement the `Fn<A>`
    /// family of traits where `A` is given by the signature of the type.
    fn consider_builtin_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        kind: ty::ClosureKind,
    ) -> QueryResult<'tcx>;

    /// `Tuple` is implemented if the `Self` type is a tuple.
    fn consider_builtin_tuple_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// `Pointee` is always implemented.
    ///
    /// See the projection implementation for the `Metadata` types for all of
    /// the built-in types. For structs, the metadata type is given by the struct
    /// tail.
    fn consider_builtin_pointee_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// A coroutine (that comes from an `async` desugaring) is known to implement
    /// `Future<Output = O>`, where `O` is given by the coroutine's return type
    /// that was computed during type-checking.
    fn consider_builtin_future_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// A coroutine (that doesn't come from an `async` desugaring) is known to
    /// implement `Coroutine<R, Yield = Y, Return = O>`, given the resume, yield,
    /// and return types of the coroutine computed during type-checking.
    fn consider_builtin_coroutine_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    fn consider_builtin_discriminant_kind_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    fn consider_builtin_destruct_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    fn consider_builtin_transmute_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// Consider (possibly several) candidates to upcast or unsize a type to another
    /// type, excluding the coercion of a sized type into a `dyn Trait`.
    ///
    /// We return the `BuiltinImplSource` for each candidate as it is needed
    /// for unsize coercion in hir typeck and because it is difficult to
    /// otherwise recompute this for codegen. This is a bit of a mess but the
    /// easiest way to maintain the existing behavior for now.
    fn consider_structural_builtin_unsize_candidates(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> Vec<(CanonicalResponse<'tcx>, BuiltinImplSource)>;

    /// Consider the `Unsize` candidate corresponding to coercing a sized type
    /// into a `dyn Trait`.
    ///
    /// This is computed separately from the rest of the `Unsize` candidates
    /// since it is only done once per self type, and not once per
    /// *normalization step* (in `assemble_candidates_via_self_ty`).
    fn consider_unsize_to_dyn_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn assemble_and_evaluate_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
    ) -> Vec<Candidate<'tcx>> {
        debug_assert_eq!(goal, self.resolve_vars_if_possible(goal));
        if let Some(ambig) = self.assemble_self_ty_infer_ambiguity_response(goal) {
            return ambig;
        }

        let mut candidates = self.assemble_candidates_via_self_ty(goal, 0);

        self.assemble_unsize_to_dyn_candidate(goal, &mut candidates);

        self.assemble_blanket_impl_candidates(goal, &mut candidates);

        self.assemble_param_env_candidates(goal, &mut candidates);

        self.assemble_coherence_unknowable_candidates(goal, &mut candidates);

        candidates
    }

    /// `?0: Trait` is ambiguous, because it may be satisfied via a builtin rule,
    /// object bound, alias bound, etc. We are unable to determine this until we can at
    /// least structurally resolve the type one layer.
    ///
    /// It would also require us to consider all impls of the trait, which is both pretty
    /// bad for perf and would also constrain the self type if there is just a single impl.
    fn assemble_self_ty_infer_ambiguity_response<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
    ) -> Option<Vec<Candidate<'tcx>>> {
        goal.predicate.self_ty().is_ty_var().then(|| {
            vec![Candidate {
                source: CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
                result: self
                    .evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                    .unwrap(),
            }]
        })
    }

    /// Assemble candidates which apply to the self type. This only looks at candidate which
    /// apply to the specific self type and ignores all others.
    ///
    /// Returns `None` if the self type is still ambiguous.
    fn assemble_candidates_via_self_ty<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        num_steps: usize,
    ) -> Vec<Candidate<'tcx>> {
        debug_assert_eq!(goal, self.resolve_vars_if_possible(goal));
        if let Some(ambig) = self.assemble_self_ty_infer_ambiguity_response(goal) {
            return ambig;
        }

        let mut candidates = Vec::new();

        self.assemble_non_blanket_impl_candidates(goal, &mut candidates);

        self.assemble_builtin_impl_candidates(goal, &mut candidates);

        self.assemble_alias_bound_candidates(goal, &mut candidates);

        self.assemble_object_bound_candidates(goal, &mut candidates);

        self.assemble_candidates_after_normalizing_self_ty(goal, &mut candidates, num_steps);
        candidates
    }

    /// If the self type of a goal is an alias we first try to normalize the self type
    /// and compute the candidates for the normalized self type in case that succeeds.
    ///
    /// These candidates are used in addition to the ones with the alias as a self type.
    /// We do this to simplify both builtin candidates and for better performance.
    ///
    /// We generate the builtin candidates on the fly by looking at the self type, e.g.
    /// add `FnPtr` candidates if the self type is a function pointer. Handling builtin
    /// candidates while the self type is still an alias seems difficult. This is similar
    /// to `try_structurally_resolve_type` during hir typeck (FIXME once implemented).
    ///
    /// Looking at all impls for some trait goal is prohibitively expensive. We therefore
    /// only look at implementations with a matching self type. Because of this function,
    /// we can avoid looking at all existing impls if the self type is an alias.
    #[instrument(level = "debug", skip_all)]
    fn assemble_candidates_after_normalizing_self_ty<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
        num_steps: usize,
    ) {
        let tcx = self.tcx();
        let &ty::Alias(_, projection_ty) = goal.predicate.self_ty().kind() else { return };

        candidates.extend(self.probe(|_| ProbeKind::NormalizedSelfTyAssembly).enter(|ecx| {
            if num_steps < ecx.local_overflow_limit() {
                let normalized_ty = ecx.next_ty_infer();
                let normalizes_to_goal = goal.with(
                    tcx,
                    ty::ProjectionPredicate { projection_ty, term: normalized_ty.into() },
                );
                ecx.add_goal(normalizes_to_goal);
                if let Err(NoSolution) = ecx.try_evaluate_added_goals() {
                    debug!("self type normalization failed");
                    return vec![];
                }
                let normalized_ty = ecx.resolve_vars_if_possible(normalized_ty);
                debug!(?normalized_ty, "self type normalized");
                // NOTE: Alternatively we could call `evaluate_goal` here and only
                // have a `Normalized` candidate. This doesn't work as long as we
                // use `CandidateSource` in winnowing.
                let goal = goal.with(tcx, goal.predicate.with_self_ty(tcx, normalized_ty));
                ecx.assemble_candidates_via_self_ty(goal, num_steps + 1)
            } else {
                match ecx.evaluate_added_goals_and_make_canonical_response(Certainty::OVERFLOW) {
                    Ok(result) => vec![Candidate {
                        source: CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
                        result,
                    }],
                    Err(NoSolution) => vec![],
                }
            }
        }));
    }

    #[instrument(level = "debug", skip_all)]
    fn assemble_non_blanket_impl_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        let self_ty = goal.predicate.self_ty();
        let trait_impls = tcx.trait_impls_of(goal.predicate.trait_def_id(tcx));
        let mut consider_impls_for_simplified_type = |simp| {
            if let Some(impls_for_type) = trait_impls.non_blanket_impls().get(&simp) {
                for &impl_def_id in impls_for_type {
                    match G::consider_impl_candidate(self, goal, impl_def_id) {
                        Ok(result) => candidates
                            .push(Candidate { source: CandidateSource::Impl(impl_def_id), result }),
                        Err(NoSolution) => (),
                    }
                }
            }
        };

        match self_ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_)
            | ty::Dynamic(_, _, _)
            | ty::Closure(_, _)
            | ty::Coroutine(_, _, _)
            | ty::Never
            | ty::Tuple(_) => {
                let simp =
                    fast_reject::simplify_type(tcx, self_ty, TreatParams::ForLookup).unwrap();
                consider_impls_for_simplified_type(simp);
            }

            // HACK: For integer and float variables we have to manually look at all impls
            // which have some integer or float as a self type.
            ty::Infer(ty::IntVar(_)) => {
                use ty::IntTy::*;
                use ty::UintTy::*;
                // This causes a compiler error if any new integer kinds are added.
                let (I8 | I16 | I32 | I64 | I128 | Isize): ty::IntTy;
                let (U8 | U16 | U32 | U64 | U128 | Usize): ty::UintTy;
                let possible_integers = [
                    // signed integers
                    SimplifiedType::Int(I8),
                    SimplifiedType::Int(I16),
                    SimplifiedType::Int(I32),
                    SimplifiedType::Int(I64),
                    SimplifiedType::Int(I128),
                    SimplifiedType::Int(Isize),
                    // unsigned integers
                    SimplifiedType::Uint(U8),
                    SimplifiedType::Uint(U16),
                    SimplifiedType::Uint(U32),
                    SimplifiedType::Uint(U64),
                    SimplifiedType::Uint(U128),
                    SimplifiedType::Uint(Usize),
                ];
                for simp in possible_integers {
                    consider_impls_for_simplified_type(simp);
                }
            }

            ty::Infer(ty::FloatVar(_)) => {
                // This causes a compiler error if any new float kinds are added.
                let (ty::FloatTy::F32 | ty::FloatTy::F64);
                let possible_floats = [
                    SimplifiedType::Float(ty::FloatTy::F32),
                    SimplifiedType::Float(ty::FloatTy::F64),
                ];

                for simp in possible_floats {
                    consider_impls_for_simplified_type(simp);
                }
            }

            // The only traits applying to aliases and placeholders are blanket impls.
            //
            // Impls which apply to an alias after normalization are handled by
            // `assemble_candidates_after_normalizing_self_ty`.
            ty::Alias(_, _) | ty::Placeholder(..) | ty::Error(_) => (),

            // FIXME: These should ideally not exist as a self type. It would be nice for
            // the builtin auto trait impls of coroutines to instead directly recurse
            // into the witness.
            ty::CoroutineWitness(..) => (),

            // These variants should not exist as a self type.
            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_))
            | ty::Param(_)
            | ty::Bound(_, _) => bug!("unexpected self type: {self_ty}"),
        }
    }

    fn assemble_unsize_to_dyn_candidate<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        if tcx.lang_items().unsize_trait() == Some(goal.predicate.trait_def_id(tcx)) {
            match G::consider_unsize_to_dyn_candidate(self, goal) {
                Ok(result) => candidates.push(Candidate {
                    source: CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
                    result,
                }),
                Err(NoSolution) => (),
            }
        }
    }

    fn assemble_blanket_impl_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        let trait_impls = tcx.trait_impls_of(goal.predicate.trait_def_id(tcx));
        for &impl_def_id in trait_impls.blanket_impls() {
            match G::consider_impl_candidate(self, goal, impl_def_id) {
                Ok(result) => candidates
                    .push(Candidate { source: CandidateSource::Impl(impl_def_id), result }),
                Err(NoSolution) => (),
            }
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn assemble_builtin_impl_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        let lang_items = tcx.lang_items();
        let trait_def_id = goal.predicate.trait_def_id(tcx);

        // N.B. When assembling built-in candidates for lang items that are also
        // `auto` traits, then the auto trait candidate that is assembled in
        // `consider_auto_trait_candidate` MUST be disqualified to remain sound.
        //
        // Instead of adding the logic here, it's a better idea to add it in
        // `EvalCtxt::disqualify_auto_trait_candidate_due_to_possible_impl` in
        // `solve::trait_goals` instead.
        let result = if let Err(guar) = goal.predicate.error_reported() {
            G::consider_error_guaranteed_candidate(self, guar)
        } else if tcx.trait_is_auto(trait_def_id) {
            G::consider_auto_trait_candidate(self, goal)
        } else if tcx.trait_is_alias(trait_def_id) {
            G::consider_trait_alias_candidate(self, goal)
        } else if lang_items.sized_trait() == Some(trait_def_id) {
            G::consider_builtin_sized_candidate(self, goal)
        } else if lang_items.copy_trait() == Some(trait_def_id)
            || lang_items.clone_trait() == Some(trait_def_id)
        {
            G::consider_builtin_copy_clone_candidate(self, goal)
        } else if lang_items.pointer_like() == Some(trait_def_id) {
            G::consider_builtin_pointer_like_candidate(self, goal)
        } else if lang_items.fn_ptr_trait() == Some(trait_def_id) {
            G::consider_builtin_fn_ptr_trait_candidate(self, goal)
        } else if let Some(kind) = self.tcx().fn_trait_kind_from_def_id(trait_def_id) {
            G::consider_builtin_fn_trait_candidates(self, goal, kind)
        } else if lang_items.tuple_trait() == Some(trait_def_id) {
            G::consider_builtin_tuple_candidate(self, goal)
        } else if lang_items.pointee_trait() == Some(trait_def_id) {
            G::consider_builtin_pointee_candidate(self, goal)
        } else if lang_items.future_trait() == Some(trait_def_id) {
            G::consider_builtin_future_candidate(self, goal)
        } else if lang_items.gen_trait() == Some(trait_def_id) {
            G::consider_builtin_coroutine_candidate(self, goal)
        } else if lang_items.discriminant_kind_trait() == Some(trait_def_id) {
            G::consider_builtin_discriminant_kind_candidate(self, goal)
        } else if lang_items.destruct_trait() == Some(trait_def_id) {
            G::consider_builtin_destruct_candidate(self, goal)
        } else if lang_items.transmute_trait() == Some(trait_def_id) {
            G::consider_builtin_transmute_candidate(self, goal)
        } else {
            Err(NoSolution)
        };

        match result {
            Ok(result) => candidates.push(Candidate {
                source: CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
                result,
            }),
            Err(NoSolution) => (),
        }

        // There may be multiple unsize candidates for a trait with several supertraits:
        // `trait Foo: Bar<A> + Bar<B>` and `dyn Foo: Unsize<dyn Bar<_>>`
        if lang_items.unsize_trait() == Some(trait_def_id) {
            for (result, source) in G::consider_structural_builtin_unsize_candidates(self, goal) {
                candidates.push(Candidate { source: CandidateSource::BuiltinImpl(source), result });
            }
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn assemble_param_env_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        for (i, assumption) in goal.param_env.caller_bounds().iter().enumerate() {
            match G::consider_implied_clause(self, goal, assumption, []) {
                Ok(result) => {
                    candidates.push(Candidate { source: CandidateSource::ParamEnv(i), result })
                }
                Err(NoSolution) => (),
            }
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn assemble_alias_bound_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let alias_ty = match goal.predicate.self_ty().kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_)
            | ty::Dynamic(..)
            | ty::Closure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Param(_)
            | ty::Placeholder(..)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Alias(ty::Inherent, _)
            | ty::Alias(ty::Weak, _)
            | ty::Error(_) => return,
            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_))
            | ty::Bound(..) => bug!("unexpected self type for `{goal:?}`"),
            // Excluding IATs and type aliases here as they don't have meaningful item bounds.
            ty::Alias(ty::Projection | ty::Opaque, alias_ty) => alias_ty,
        };

        for assumption in
            self.tcx().item_bounds(alias_ty.def_id).instantiate(self.tcx(), alias_ty.args)
        {
            match G::consider_alias_bound_candidate(self, goal, assumption) {
                Ok(result) => {
                    candidates.push(Candidate { source: CandidateSource::AliasBound, result })
                }
                Err(NoSolution) => (),
            }
        }
    }

    /// Check that we are allowed to use an alias bound originating from the self
    /// type of this goal. This means something different depending on the self type's
    /// alias kind.
    ///
    /// * Projection: Given a goal with a self type such as `<Ty as Trait>::Assoc`,
    /// we require that the bound `Ty: Trait` can be proven using either a nested alias
    /// bound candidate, or a param-env candidate.
    ///
    /// * Opaque: The param-env must be in `Reveal::UserFacing` mode. Otherwise,
    /// the goal should be proven by using the hidden type instead.
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn validate_alias_bound_self_from_param_env<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
    ) -> QueryResult<'tcx> {
        match *goal.predicate.self_ty().kind() {
            ty::Alias(ty::Projection, projection_ty) => {
                let mut param_env_candidates = vec![];
                let self_trait_ref = projection_ty.trait_ref(self.tcx());

                if self_trait_ref.self_ty().is_ty_var() {
                    return self
                        .evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
                }

                let trait_goal: Goal<'_, ty::TraitPredicate<'tcx>> = goal.with(
                    self.tcx(),
                    ty::TraitPredicate {
                        trait_ref: self_trait_ref,
                        polarity: ty::ImplPolarity::Positive,
                    },
                );

                self.assemble_param_env_candidates(trait_goal, &mut param_env_candidates);
                // FIXME: We probably need some sort of recursion depth check here.
                // Can't come up with an example yet, though, and the worst case
                // we can have is a compiler stack overflow...
                self.assemble_alias_bound_candidates(trait_goal, &mut param_env_candidates);

                // FIXME: We must also consider alias-bound candidates for a peculiar
                // class of built-in candidates that I'll call "defaulted" built-ins.
                //
                // For example, we always know that `T: Pointee` is implemented, but
                // we do not always know what `<T as Pointee>::Metadata` actually is,
                // similar to if we had a user-defined impl with a `default type ...`.
                // For these traits, since we're not able to always normalize their
                // associated types to a concrete type, we must consider their alias bounds
                // instead, so we can prove bounds such as `<T as Pointee>::Metadata: Copy`.
                self.assemble_alias_bound_candidates_for_builtin_impl_default_items(
                    trait_goal,
                    &mut param_env_candidates,
                );

                self.merge_candidates(param_env_candidates)
            }
            ty::Alias(ty::Opaque, _opaque_ty) => match goal.param_env.reveal() {
                Reveal::UserFacing => {
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
                }
                Reveal::All => return Err(NoSolution),
            },
            _ => bug!("only expected to be called on alias tys"),
        }
    }

    /// Assemble a subset of builtin impl candidates for a class of candidates called
    /// "defaulted" built-in traits.
    ///
    /// For example, we always know that `T: Pointee` is implemented, but we do not
    /// always know what `<T as Pointee>::Metadata` actually is! See the comment in
    /// [`EvalCtxt::validate_alias_bound_self_from_param_env`] for more detail.
    #[instrument(level = "debug", skip_all)]
    fn assemble_alias_bound_candidates_for_builtin_impl_default_items<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let lang_items = self.tcx().lang_items();
        let trait_def_id = goal.predicate.trait_def_id(self.tcx());

        // You probably shouldn't add anything to this list unless you
        // know what you're doing.
        let result = if lang_items.pointee_trait() == Some(trait_def_id) {
            G::consider_builtin_pointee_candidate(self, goal)
        } else if lang_items.discriminant_kind_trait() == Some(trait_def_id) {
            G::consider_builtin_discriminant_kind_candidate(self, goal)
        } else {
            Err(NoSolution)
        };

        match result {
            Ok(result) => candidates.push(Candidate {
                source: CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
                result,
            }),
            Err(NoSolution) => (),
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn assemble_object_bound_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        if !tcx.trait_def(goal.predicate.trait_def_id(tcx)).implement_via_object {
            return;
        }

        let self_ty = goal.predicate.self_ty();
        let bounds = match *self_ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_)
            | ty::Alias(..)
            | ty::Closure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Param(_)
            | ty::Placeholder(..)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Error(_) => return,
            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_))
            | ty::Bound(..) => bug!("unexpected self type for `{goal:?}`"),
            ty::Dynamic(bounds, ..) => bounds,
        };

        // Do not consider built-in object impls for non-object-safe types.
        if bounds.principal_def_id().is_some_and(|def_id| !tcx.check_is_object_safe(def_id)) {
            return;
        }

        // Consider all of the auto-trait and projection bounds, which don't
        // need to be recorded as a `BuiltinImplSource::Object` since they don't
        // really have a vtable base...
        for bound in bounds {
            match bound.skip_binder() {
                ty::ExistentialPredicate::Trait(_) => {
                    // Skip principal
                }
                ty::ExistentialPredicate::Projection(_)
                | ty::ExistentialPredicate::AutoTrait(_) => {
                    match G::consider_object_bound_candidate(
                        self,
                        goal,
                        bound.with_self_ty(tcx, self_ty),
                    ) {
                        Ok(result) => candidates.push(Candidate {
                            source: CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
                            result,
                        }),
                        Err(NoSolution) => (),
                    }
                }
            }
        }

        // FIXME: We only need to do *any* of this if we're considering a trait goal,
        // since we don't need to look at any supertrait or anything if we are doing
        // a projection goal.
        if let Some(principal) = bounds.principal() {
            let principal_trait_ref = principal.with_self_ty(tcx, self_ty);
            self.walk_vtable(principal_trait_ref, |ecx, assumption, vtable_base, _| {
                match G::consider_object_bound_candidate(ecx, goal, assumption.to_predicate(tcx)) {
                    Ok(result) => candidates.push(Candidate {
                        source: CandidateSource::BuiltinImpl(BuiltinImplSource::Object {
                            vtable_base,
                        }),
                        result,
                    }),
                    Err(NoSolution) => (),
                }
            });
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn assemble_coherence_unknowable_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        match self.solver_mode() {
            SolverMode::Normal => return,
            SolverMode::Coherence => {}
        };

        let result = self.probe_misc_candidate("coherence unknowable").enter(|ecx| {
            let trait_ref = goal.predicate.trait_ref(tcx);

            #[derive(Debug)]
            enum FailureKind {
                Overflow,
                NoSolution(NoSolution),
            }
            let lazily_normalize_ty = |ty| match ecx.try_normalize_ty(goal.param_env, ty) {
                Ok(Some(ty)) => Ok(ty),
                Ok(None) => Err(FailureKind::Overflow),
                Err(e) => Err(FailureKind::NoSolution(e)),
            };

            match coherence::trait_ref_is_knowable(tcx, trait_ref, lazily_normalize_ty) {
                Err(FailureKind::Overflow) => {
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::OVERFLOW)
                }
                Err(FailureKind::NoSolution(NoSolution)) | Ok(Ok(())) => Err(NoSolution),
                Ok(Err(_)) => {
                    ecx.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                }
            }
        });

        match result {
            Ok(result) => candidates.push(Candidate {
                source: CandidateSource::BuiltinImpl(BuiltinImplSource::Misc),
                result,
            }),
            Err(NoSolution) => {}
        }
    }

    /// If there are multiple ways to prove a trait or projection goal, we have
    /// to somehow try to merge the candidates into one. If that fails, we return
    /// ambiguity.
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn merge_candidates(
        &mut self,
        mut candidates: Vec<Candidate<'tcx>>,
    ) -> QueryResult<'tcx> {
        // First try merging all candidates. This is complete and fully sound.
        let responses = candidates.iter().map(|c| c.result).collect::<Vec<_>>();
        if let Some(result) = self.try_merge_responses(&responses) {
            return Ok(result);
        }

        // We then check whether we should prioritize `ParamEnv` candidates.
        //
        // Doing so is incomplete and would therefore be unsound during coherence.
        match self.solver_mode() {
            SolverMode::Coherence => (),
            // Prioritize `ParamEnv` candidates only if they do not guide inference.
            //
            // This is still incomplete as we may add incorrect region bounds.
            SolverMode::Normal => {
                let param_env_responses = candidates
                    .iter()
                    .filter(|c| {
                        matches!(
                            c.source,
                            CandidateSource::ParamEnv(_) | CandidateSource::AliasBound
                        )
                    })
                    .map(|c| c.result)
                    .collect::<Vec<_>>();
                if let Some(result) = self.try_merge_responses(&param_env_responses) {
                    // We strongly prefer alias and param-env bounds here, even if they affect inference.
                    // See https://github.com/rust-lang/trait-system-refactor-initiative/issues/11.
                    return Ok(result);
                }
            }
        }
        self.flounder(&responses)
    }
}

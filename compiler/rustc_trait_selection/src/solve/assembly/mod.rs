//! Code shared by trait and projection goals for candidate assembly.

use super::{EvalCtxt, SolverMode};
use crate::solve::GoalSource;
use crate::traits::coherence;
use rustc_hir::def_id::DefId;
use rustc_infer::traits::query::NoSolution;
use rustc_middle::traits::solve::inspect::ProbeKind;
use rustc_middle::traits::solve::{
    CandidateSource, CanonicalResponse, Certainty, Goal, MaybeCause, QueryResult,
};
use rustc_middle::traits::BuiltinImplSource;
use rustc_middle::ty::fast_reject::{SimplifiedType, TreatParams};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{fast_reject, TypeFoldable};
use rustc_middle::ty::{ToPredicate, TypeVisitableExt};
use rustc_span::{ErrorGuaranteed, DUMMY_SP};
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
            // FIXME(-Znext-solver=coinductive): check whether this should be
            // `GoalSource::ImplWhereBound` for any caller.
            ecx.add_goals(GoalSource::Misc, requirements);
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
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
            // FIXME(-Znext-solver=coinductive): Should this be `GoalSource::ImplWhereBound`?
            ecx.add_goals(
                GoalSource::Misc,
                structural_traits::predicates_for_object_candidate(
                    ecx,
                    goal.param_env,
                    goal.predicate.trait_ref(tcx),
                    bounds,
                ),
            );
            ecx.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
        })
    }

    fn consider_impl_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        impl_def_id: DefId,
    ) -> Result<Candidate<'tcx>, NoSolution>;

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

    /// An async closure is known to implement the `AsyncFn<A>` family of traits
    /// where `A` is given by the signature of the type.
    fn consider_builtin_async_fn_trait_candidates(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
        kind: ty::ClosureKind,
    ) -> QueryResult<'tcx>;

    /// Compute the built-in logic of the `AsyncFnKindHelper` helper trait, which
    /// is used internally to delay computation for async closures until after
    /// upvar analysis is performed in HIR typeck.
    fn consider_builtin_async_fn_kind_helper_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
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

    /// A coroutine (that comes from a `gen` desugaring) is known to implement
    /// `Iterator<Item = O>`, where `O` is given by the generator's yield type
    /// that was computed during type-checking.
    fn consider_builtin_iterator_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    fn consider_builtin_async_iterator_candidate(
        ecx: &mut EvalCtxt<'_, 'tcx>,
        goal: Goal<'tcx, Self>,
    ) -> QueryResult<'tcx>;

    /// A coroutine (that doesn't come from an `async` or `gen` desugaring) is known to
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
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn assemble_and_evaluate_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
    ) -> Vec<Candidate<'tcx>> {
        let Ok(normalized_self_ty) =
            self.structurally_normalize_ty(goal.param_env, goal.predicate.self_ty())
        else {
            return vec![];
        };

        if normalized_self_ty.is_ty_var() {
            debug!("self type has been normalized to infer");
            return self.forced_ambiguity(MaybeCause::Ambiguity);
        }

        let goal =
            goal.with(self.tcx(), goal.predicate.with_self_ty(self.tcx(), normalized_self_ty));
        debug_assert_eq!(goal, self.resolve_vars_if_possible(goal));

        let mut candidates = vec![];

        self.assemble_non_blanket_impl_candidates(goal, &mut candidates);

        self.assemble_builtin_impl_candidates(goal, &mut candidates);

        self.assemble_alias_bound_candidates(goal, &mut candidates);

        self.assemble_object_bound_candidates(goal, &mut candidates);

        self.assemble_blanket_impl_candidates(goal, &mut candidates);

        self.assemble_param_env_candidates(goal, &mut candidates);

        match self.solver_mode() {
            SolverMode::Normal => self.discard_impls_shadowed_by_env(goal, &mut candidates),
            SolverMode::Coherence => {
                self.assemble_coherence_unknowable_candidates(goal, &mut candidates)
            }
        }

        candidates
    }

    fn forced_ambiguity(&mut self, cause: MaybeCause) -> Vec<Candidate<'tcx>> {
        let source = CandidateSource::BuiltinImpl(BuiltinImplSource::Misc);
        let certainty = Certainty::Maybe(cause);
        let result = self.evaluate_added_goals_and_make_canonical_response(certainty).unwrap();
        let mut dummy_probe = self.inspect.new_probe();
        dummy_probe.probe_kind(ProbeKind::TraitCandidate { source, result: Ok(result) });
        self.inspect.finish_probe(dummy_probe);
        vec![Candidate { source, result }]
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
                    // For every `default impl`, there's always a non-default `impl`
                    // that will *also* apply. There's no reason to register a candidate
                    // for this impl, since it is *not* proof that the trait goal holds.
                    if tcx.defaultness(impl_def_id).is_default() {
                        return;
                    }

                    match G::consider_impl_candidate(self, goal, impl_def_id) {
                        Ok(candidate) => candidates.push(candidate),
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
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
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
                let (ty::FloatTy::F16 | ty::FloatTy::F32 | ty::FloatTy::F64 | ty::FloatTy::F128);
                let possible_floats = [
                    SimplifiedType::Float(ty::FloatTy::F16),
                    SimplifiedType::Float(ty::FloatTy::F32),
                    SimplifiedType::Float(ty::FloatTy::F64),
                    SimplifiedType::Float(ty::FloatTy::F128),
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

    #[instrument(level = "debug", skip_all)]
    fn assemble_blanket_impl_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        let trait_impls = tcx.trait_impls_of(goal.predicate.trait_def_id(tcx));
        for &impl_def_id in trait_impls.blanket_impls() {
            // For every `default impl`, there's always a non-default `impl`
            // that will *also* apply. There's no reason to register a candidate
            // for this impl, since it is *not* proof that the trait goal holds.
            if tcx.defaultness(impl_def_id).is_default() {
                return;
            }

            match G::consider_impl_candidate(self, goal, impl_def_id) {
                Ok(candidate) => candidates.push(candidate),
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
        } else if let Some(kind) = self.tcx().async_fn_trait_kind_from_def_id(trait_def_id) {
            G::consider_builtin_async_fn_trait_candidates(self, goal, kind)
        } else if lang_items.async_fn_kind_helper() == Some(trait_def_id) {
            G::consider_builtin_async_fn_kind_helper_candidate(self, goal)
        } else if lang_items.tuple_trait() == Some(trait_def_id) {
            G::consider_builtin_tuple_candidate(self, goal)
        } else if lang_items.pointee_trait() == Some(trait_def_id) {
            G::consider_builtin_pointee_candidate(self, goal)
        } else if lang_items.future_trait() == Some(trait_def_id) {
            G::consider_builtin_future_candidate(self, goal)
        } else if lang_items.iterator_trait() == Some(trait_def_id) {
            G::consider_builtin_iterator_candidate(self, goal)
        } else if lang_items.async_iterator_trait() == Some(trait_def_id) {
            G::consider_builtin_async_iterator_candidate(self, goal)
        } else if lang_items.coroutine_trait() == Some(trait_def_id) {
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
        let () = self.probe(|_| ProbeKind::NormalizedSelfTyAssembly).enter(|ecx| {
            ecx.assemble_alias_bound_candidates_recur(goal.predicate.self_ty(), goal, candidates);
        });
    }

    /// For some deeply nested `<T>::A::B::C::D` rigid associated type,
    /// we should explore the item bounds for all levels, since the
    /// `associated_type_bounds` feature means that a parent associated
    /// type may carry bounds for a nested associated type.
    ///
    /// If we have a projection, check that its self type is a rigid projection.
    /// If so, continue searching by recursively calling after normalization.
    // FIXME: This may recurse infinitely, but I can't seem to trigger it without
    // hitting another overflow error something. Add a depth parameter needed later.
    fn assemble_alias_bound_candidates_recur<G: GoalKind<'tcx>>(
        &mut self,
        self_ty: Ty<'tcx>,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let (kind, alias_ty) = match *self_ty.kind() {
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
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Param(_)
            | ty::Placeholder(..)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Error(_) => return,
            ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) | ty::Bound(..) => {
                bug!("unexpected self type for `{goal:?}`")
            }

            ty::Infer(ty::TyVar(_)) => {
                // If we hit infer when normalizing the self type of an alias,
                // then bail with ambiguity. We should never encounter this on
                // the *first* iteration of this recursive function.
                if let Ok(result) =
                    self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS)
                {
                    candidates.push(Candidate { source: CandidateSource::AliasBound, result });
                }
                return;
            }

            ty::Alias(kind @ (ty::Projection | ty::Opaque), alias_ty) => (kind, alias_ty),
            ty::Alias(ty::Inherent | ty::Weak, _) => {
                self.tcx().sess.dcx().span_delayed_bug(
                    DUMMY_SP,
                    format!("could not normalize {self_ty}, it is not WF"),
                );
                return;
            }
        };

        for assumption in
            self.tcx().item_bounds(alias_ty.def_id).instantiate(self.tcx(), alias_ty.args)
        {
            match G::consider_implied_clause(self, goal, assumption, []) {
                Ok(result) => {
                    candidates.push(Candidate { source: CandidateSource::AliasBound, result });
                }
                Err(NoSolution) => {}
            }
        }

        if kind != ty::Projection {
            return;
        }

        // Recurse on the self type of the projection.
        match self.structurally_normalize_ty(goal.param_env, alias_ty.self_ty()) {
            Ok(next_self_ty) => {
                self.assemble_alias_bound_candidates_recur(next_self_ty, goal, candidates)
            }
            Err(NoSolution) => {}
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
            | ty::CoroutineClosure(..)
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

    /// In coherence we have to not only care about all impls we know about, but
    /// also consider impls which may get added in a downstream or sibling crate
    /// or which an upstream impl may add in a minor release.
    ///
    /// To do so we add an ambiguous candidate in case such an unknown impl could
    /// apply to the current goal.
    #[instrument(level = "debug", skip_all)]
    fn assemble_coherence_unknowable_candidates<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        let result = self.probe_misc_candidate("coherence unknowable").enter(|ecx| {
            let trait_ref = goal.predicate.trait_ref(tcx);
            let lazily_normalize_ty = |ty| ecx.structurally_normalize_ty(goal.param_env, ty);

            match coherence::trait_ref_is_knowable(tcx, trait_ref, lazily_normalize_ty)? {
                Ok(()) => Err(NoSolution),
                Err(_) => {
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

    /// If there's a where-bound for the current goal, do not use any impl candidates
    /// to prove the current goal. Most importantly, if there is a where-bound which does
    /// not specify any associated types, we do not allow normalizing the associated type
    /// by using an impl, even if it would apply.
    ///
    ///  <https://github.com/rust-lang/trait-system-refactor-initiative/issues/76>
    // FIXME(@lcnr): The current structure here makes me unhappy and feels ugly. idk how
    // to improve this however. However, this should make it fairly straightforward to refine
    // the filtering going forward, so it seems alright-ish for now.
    #[instrument(level = "debug", skip(self, goal))]
    fn discard_impls_shadowed_by_env<G: GoalKind<'tcx>>(
        &mut self,
        goal: Goal<'tcx, G>,
        candidates: &mut Vec<Candidate<'tcx>>,
    ) {
        let tcx = self.tcx();
        let trait_goal: Goal<'tcx, ty::TraitPredicate<'tcx>> =
            goal.with(tcx, goal.predicate.trait_ref(tcx));
        let mut trait_candidates_from_env = Vec::new();
        self.assemble_param_env_candidates(trait_goal, &mut trait_candidates_from_env);
        self.assemble_alias_bound_candidates(trait_goal, &mut trait_candidates_from_env);
        if !trait_candidates_from_env.is_empty() {
            let trait_env_result = self.merge_candidates(trait_candidates_from_env);
            match trait_env_result.unwrap().value.certainty {
                // If proving the trait goal succeeds by using the env,
                // we freely drop all impl candidates.
                //
                // FIXME(@lcnr): It feels like this could easily hide
                // a forced ambiguity candidate added earlier.
                // This feels dangerous.
                Certainty::Yes => {
                    candidates.retain(|c| match c.source {
                        CandidateSource::Impl(_) | CandidateSource::BuiltinImpl(_) => {
                            debug!(?c, "discard impl candidate");
                            false
                        }
                        CandidateSource::ParamEnv(_) | CandidateSource::AliasBound => true,
                    });
                }
                // If it is still ambiguous we instead just force the whole goal
                // to be ambig and wait for inference constraints. See
                // tests/ui/traits/next-solver/env-shadows-impls/ambig-env-no-shadow.rs
                Certainty::Maybe(cause) => {
                    debug!(?cause, "force ambiguity");
                    *candidates = self.forced_ambiguity(cause);
                }
            }
        }
    }

    /// If there are multiple ways to prove a trait or projection goal, we have
    /// to somehow try to merge the candidates into one. If that fails, we return
    /// ambiguity.
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn merge_candidates(
        &mut self,
        candidates: Vec<Candidate<'tcx>>,
    ) -> QueryResult<'tcx> {
        // First try merging all candidates. This is complete and fully sound.
        let responses = candidates.iter().map(|c| c.result).collect::<Vec<_>>();
        if let Some(result) = self.try_merge_responses(&responses) {
            return Ok(result);
        } else {
            self.flounder(&responses)
        }
    }
}

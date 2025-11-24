//! Unification and canonicalization logic.

use std::fmt;

use hir_def::{AdtId, DefWithBodyId, GenericParamId, lang_item::LangItem};
use hir_expand::name::Name;
use intern::sym;
use rustc_hash::FxHashSet;
use rustc_type_ir::{
    DebruijnIndex, InferConst, InferTy, RegionVid, TyVid, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeVisitableExt, UpcastFrom,
    inherent::{Const as _, IntoKind, Ty as _},
    solve::{Certainty, GoalSource},
};
use smallvec::SmallVec;
use triomphe::Arc;

use crate::{
    TraitEnvironment,
    db::HirDatabase,
    infer::InferenceContext,
    next_solver::{
        self, AliasTy, Binder, Canonical, ClauseKind, Const, ConstKind, DbInterner,
        ErrorGuaranteed, GenericArg, GenericArgs, Predicate, PredicateKind, Region, RegionKind,
        SolverDefId, TraitRef, Ty, TyKind, TypingMode,
        fulfill::{FulfillmentCtxt, NextSolverError},
        infer::{
            DbInternerInferExt, InferCtxt, InferOk, InferResult,
            at::ToTrace,
            snapshot::CombinedSnapshot,
            traits::{Obligation, ObligationCause, PredicateObligation},
        },
        inspect::{InspectConfig, InspectGoal, ProofTreeVisitor},
        obligation_ctxt::ObligationCtxt,
    },
    traits::{
        FnTrait, NextTraitSolveResult, next_trait_solve_canonical_in_ctxt, next_trait_solve_in_ctxt,
    },
};

impl<'db> InferenceContext<'_, 'db> {
    pub(super) fn canonicalize<T>(&mut self, t: T) -> rustc_type_ir::Canonical<DbInterner<'db>, T>
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        self.table.canonicalize(t)
    }
}

struct NestedObligationsForSelfTy<'a, 'db> {
    ctx: &'a InferenceTable<'db>,
    self_ty: TyVid,
    root_cause: &'a ObligationCause,
    obligations_for_self_ty: &'a mut SmallVec<[Obligation<'db, Predicate<'db>>; 4]>,
}

impl<'a, 'db> ProofTreeVisitor<'db> for NestedObligationsForSelfTy<'a, 'db> {
    type Result = ();

    fn config(&self) -> InspectConfig {
        // Using an intentionally low depth to minimize the chance of future
        // breaking changes in case we adapt the approach later on. This also
        // avoids any hangs for exponentially growing proof trees.
        InspectConfig { max_depth: 5 }
    }

    fn visit_goal(&mut self, inspect_goal: &InspectGoal<'_, 'db>) {
        // No need to walk into goal subtrees that certainly hold, since they
        // wouldn't then be stalled on an infer var.
        if inspect_goal.result() == Ok(Certainty::Yes) {
            return;
        }

        let db = self.ctx.interner();
        let goal = inspect_goal.goal();
        if self.ctx.predicate_has_self_ty(goal.predicate, self.self_ty)
            // We do not push the instantiated forms of goals as it would cause any
            // aliases referencing bound vars to go from having escaping bound vars to
            // being able to be normalized to an inference variable.
            //
            // This is mostly just a hack as arbitrary nested goals could still contain
            // such aliases while having a different `GoalSource`. Closure signature inference
            // however can't really handle *every* higher ranked `Fn` goal also being present
            // in the form of `?c: Fn<(<?x as Trait<'!a>>::Assoc)`.
            //
            // This also just better matches the behaviour of the old solver where we do not
            // encounter instantiated forms of goals, only nested goals that referred to bound
            // vars from instantiated goals.
            && !matches!(inspect_goal.source(), GoalSource::InstantiateHigherRanked)
        {
            self.obligations_for_self_ty.push(Obligation::new(
                db,
                self.root_cause.clone(),
                goal.param_env,
                goal.predicate,
            ));
        }

        // If there's a unique way to prove a given goal, recurse into
        // that candidate. This means that for `impl<F: FnOnce(u32)> Trait<F> for () {}`
        // and a `(): Trait<?0>` goal we recurse into the impl and look at
        // the nested `?0: FnOnce(u32)` goal.
        if let Some(candidate) = inspect_goal.unique_applicable_candidate() {
            candidate.visit_nested_no_probe(self)
        }
    }
}

/// Check if types unify.
///
/// Note that we consider placeholder types to unify with everything.
/// This means that there may be some unresolved goals that actually set bounds for the placeholder
/// type for the types to unify. For example `Option<T>` and `Option<U>` unify although there is
/// unresolved goal `T = U`.
pub fn could_unify<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    tys: &Canonical<'db, (Ty<'db>, Ty<'db>)>,
) -> bool {
    could_unify_impl(db, env, tys, |ctxt| ctxt.try_evaluate_obligations())
}

/// Check if types unify eagerly making sure there are no unresolved goals.
///
/// This means that placeholder types are not considered to unify if there are any bounds set on
/// them. For example `Option<T>` and `Option<U>` do not unify as we cannot show that `T = U`
pub fn could_unify_deeply<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    tys: &Canonical<'db, (Ty<'db>, Ty<'db>)>,
) -> bool {
    could_unify_impl(db, env, tys, |ctxt| ctxt.evaluate_obligations_error_on_ambiguity())
}

fn could_unify_impl<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    tys: &Canonical<'db, (Ty<'db>, Ty<'db>)>,
    select: for<'a> fn(&mut ObligationCtxt<'a, 'db>) -> Vec<NextSolverError<'db>>,
) -> bool {
    let interner = DbInterner::new_with(db, Some(env.krate), env.block);
    let infcx = interner.infer_ctxt().build(TypingMode::PostAnalysis);
    let cause = ObligationCause::dummy();
    let at = infcx.at(&cause, env.env);
    let ((ty1_with_vars, ty2_with_vars), _) = infcx.instantiate_canonical(tys);
    let mut ctxt = ObligationCtxt::new(&infcx);
    let can_unify = at
        .eq(ty1_with_vars, ty2_with_vars)
        .map(|infer_ok| ctxt.register_infer_ok_obligations(infer_ok))
        .is_ok();
    can_unify && select(&mut ctxt).is_empty()
}

#[derive(Clone)]
pub(crate) struct InferenceTable<'db> {
    pub(crate) db: &'db dyn HirDatabase,
    pub(crate) trait_env: Arc<TraitEnvironment<'db>>,
    pub(crate) infer_ctxt: InferCtxt<'db>,
    pub(super) fulfillment_cx: FulfillmentCtxt<'db>,
    pub(super) diverging_type_vars: FxHashSet<Ty<'db>>,
}

pub(crate) struct InferenceTableSnapshot<'db> {
    ctxt_snapshot: CombinedSnapshot,
    obligations: FulfillmentCtxt<'db>,
}

impl<'db> InferenceTable<'db> {
    /// Inside hir-ty you should use this for inference only, and always pass `owner`.
    /// Outside it, always pass `owner = None`.
    pub(crate) fn new(
        db: &'db dyn HirDatabase,
        trait_env: Arc<TraitEnvironment<'db>>,
        owner: Option<DefWithBodyId>,
    ) -> Self {
        let interner = DbInterner::new_with(db, Some(trait_env.krate), trait_env.block);
        let typing_mode = match owner {
            Some(owner) => TypingMode::typeck_for_body(interner, owner.into()),
            // IDE things wants to reveal opaque types.
            None => TypingMode::PostAnalysis,
        };
        let infer_ctxt = interner.infer_ctxt().build(typing_mode);
        InferenceTable {
            db,
            trait_env,
            fulfillment_cx: FulfillmentCtxt::new(&infer_ctxt),
            infer_ctxt,
            diverging_type_vars: FxHashSet::default(),
        }
    }

    #[inline]
    pub(crate) fn interner(&self) -> DbInterner<'db> {
        self.infer_ctxt.interner
    }

    pub(crate) fn type_is_copy_modulo_regions(&self, ty: Ty<'db>) -> bool {
        self.infer_ctxt.type_is_copy_modulo_regions(self.trait_env.env, ty)
    }

    pub(crate) fn type_var_is_sized(&self, self_ty: TyVid) -> bool {
        let Some(sized_did) = LangItem::Sized.resolve_trait(self.db, self.trait_env.krate) else {
            return true;
        };
        self.obligations_for_self_ty(self_ty).into_iter().any(|obligation| {
            match obligation.predicate.kind().skip_binder() {
                PredicateKind::Clause(ClauseKind::Trait(data)) => data.def_id().0 == sized_did,
                _ => false,
            }
        })
    }

    pub(super) fn obligations_for_self_ty(
        &self,
        self_ty: TyVid,
    ) -> SmallVec<[Obligation<'db, Predicate<'db>>; 4]> {
        let obligations = self.fulfillment_cx.pending_obligations();
        let mut obligations_for_self_ty = SmallVec::new();
        for obligation in obligations {
            let mut visitor = NestedObligationsForSelfTy {
                ctx: self,
                self_ty,
                obligations_for_self_ty: &mut obligations_for_self_ty,
                root_cause: &obligation.cause,
            };

            let goal = obligation.as_goal();
            self.infer_ctxt.visit_proof_tree(goal, &mut visitor);
        }

        obligations_for_self_ty.retain_mut(|obligation| {
            obligation.predicate = self.infer_ctxt.resolve_vars_if_possible(obligation.predicate);
            !obligation.predicate.has_placeholders()
        });
        obligations_for_self_ty
    }

    fn predicate_has_self_ty(&self, predicate: Predicate<'db>, expected_vid: TyVid) -> bool {
        match predicate.kind().skip_binder() {
            PredicateKind::Clause(ClauseKind::Trait(data)) => {
                self.type_matches_expected_vid(expected_vid, data.self_ty())
            }
            PredicateKind::Clause(ClauseKind::Projection(data)) => {
                self.type_matches_expected_vid(expected_vid, data.projection_term.self_ty())
            }
            PredicateKind::Clause(ClauseKind::ConstArgHasType(..))
            | PredicateKind::Subtype(..)
            | PredicateKind::Coerce(..)
            | PredicateKind::Clause(ClauseKind::RegionOutlives(..))
            | PredicateKind::Clause(ClauseKind::TypeOutlives(..))
            | PredicateKind::Clause(ClauseKind::WellFormed(..))
            | PredicateKind::DynCompatible(..)
            | PredicateKind::NormalizesTo(..)
            | PredicateKind::AliasRelate(..)
            | PredicateKind::Clause(ClauseKind::ConstEvaluatable(..))
            | PredicateKind::ConstEquate(..)
            | PredicateKind::Clause(ClauseKind::HostEffect(..))
            | PredicateKind::Clause(ClauseKind::UnstableFeature(_))
            | PredicateKind::Ambiguous => false,
        }
    }

    fn type_matches_expected_vid(&self, expected_vid: TyVid, ty: Ty<'db>) -> bool {
        let ty = self.shallow_resolve(ty);

        match ty.kind() {
            TyKind::Infer(rustc_type_ir::TyVar(found_vid)) => {
                self.infer_ctxt.root_var(expected_vid) == self.infer_ctxt.root_var(found_vid)
            }
            _ => false,
        }
    }

    pub(super) fn set_diverging(&mut self, ty: Ty<'db>) {
        self.diverging_type_vars.insert(ty);
    }

    pub(crate) fn canonicalize<T>(&mut self, t: T) -> rustc_type_ir::Canonical<DbInterner<'db>, T>
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        // try to resolve obligations before canonicalizing, since this might
        // result in new knowledge about variables
        self.select_obligations_where_possible();
        self.infer_ctxt.canonicalize_response(t)
    }

    // FIXME: We should get rid of this method. We cannot deeply normalize during inference, only when finishing.
    // Inference should use shallow normalization (`try_structurally_resolve_type()`) only, when needed.
    pub(crate) fn normalize_associated_types_in<T>(&mut self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>> + Clone,
    {
        let ty = self.resolve_vars_with_obligations(ty);
        self.infer_ctxt
            .at(&ObligationCause::new(), self.trait_env.env)
            .deeply_normalize(ty.clone())
            .unwrap_or(ty)
    }

    /// Works almost same as [`Self::normalize_associated_types_in`], but this also resolves shallow
    /// the inference variables
    pub(crate) fn eagerly_normalize_and_resolve_shallow_in<T>(&mut self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        let ty = self.resolve_vars_with_obligations(ty);
        let ty = self.normalize_associated_types_in(ty);
        self.resolve_vars_with_obligations(ty)
    }

    pub(crate) fn normalize_alias_ty(&mut self, alias: Ty<'db>) -> Ty<'db> {
        self.infer_ctxt
            .at(&ObligationCause::new(), self.trait_env.env)
            .structurally_normalize_ty(alias, &mut self.fulfillment_cx)
            .unwrap_or(alias)
    }

    pub(crate) fn next_ty_var(&mut self) -> Ty<'db> {
        self.infer_ctxt.next_ty_var()
    }

    pub(crate) fn next_const_var(&mut self) -> Const<'db> {
        self.infer_ctxt.next_const_var()
    }

    pub(crate) fn next_int_var(&mut self) -> Ty<'db> {
        self.infer_ctxt.next_int_var()
    }

    pub(crate) fn next_float_var(&mut self) -> Ty<'db> {
        self.infer_ctxt.next_float_var()
    }

    pub(crate) fn new_maybe_never_var(&mut self) -> Ty<'db> {
        let var = self.next_ty_var();
        self.set_diverging(var);
        var
    }

    pub(crate) fn next_region_var(&mut self) -> Region<'db> {
        self.infer_ctxt.next_region_var()
    }

    pub(crate) fn next_var_for_param(&mut self, id: GenericParamId) -> GenericArg<'db> {
        match id {
            GenericParamId::TypeParamId(_) => self.next_ty_var().into(),
            GenericParamId::ConstParamId(_) => self.next_const_var().into(),
            GenericParamId::LifetimeParamId(_) => self.next_region_var().into(),
        }
    }

    pub(crate) fn resolve_with_fallback<T>(
        &mut self,
        t: T,
        fallback_ty: &mut dyn FnMut(DebruijnIndex, InferTy) -> Ty<'db>,
        fallback_const: &mut dyn FnMut(DebruijnIndex, InferConst) -> Const<'db>,
        fallback_region: &mut dyn FnMut(DebruijnIndex, RegionVid) -> Region<'db>,
    ) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        struct Resolver<'a, 'db> {
            table: &'a mut InferenceTable<'db>,
            binder: DebruijnIndex,
            fallback_ty: &'a mut dyn FnMut(DebruijnIndex, InferTy) -> Ty<'db>,
            fallback_const: &'a mut dyn FnMut(DebruijnIndex, InferConst) -> Const<'db>,
            fallback_region: &'a mut dyn FnMut(DebruijnIndex, RegionVid) -> Region<'db>,
        }

        impl<'db> TypeFolder<DbInterner<'db>> for Resolver<'_, 'db> {
            fn cx(&self) -> DbInterner<'db> {
                self.table.interner()
            }

            fn fold_binder<T>(&mut self, t: Binder<'db, T>) -> Binder<'db, T>
            where
                T: TypeFoldable<DbInterner<'db>>,
            {
                self.binder.shift_in(1);
                let result = t.super_fold_with(self);
                self.binder.shift_out(1);
                result
            }

            fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
                if !t.has_infer() {
                    return t;
                }

                if let TyKind::Infer(infer) = t.kind() {
                    (self.fallback_ty)(self.binder, infer)
                } else {
                    t.super_fold_with(self)
                }
            }

            fn fold_const(&mut self, c: Const<'db>) -> Const<'db> {
                if !c.has_infer() {
                    return c;
                }

                if let ConstKind::Infer(infer) = c.kind() {
                    (self.fallback_const)(self.binder, infer)
                } else {
                    c.super_fold_with(self)
                }
            }

            fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
                if let RegionKind::ReVar(infer) = r.kind() {
                    (self.fallback_region)(self.binder, infer)
                } else {
                    r
                }
            }
        }

        t.fold_with(&mut Resolver {
            table: self,
            binder: DebruijnIndex::ZERO,
            fallback_ty,
            fallback_const,
            fallback_region,
        })
    }

    pub(crate) fn instantiate_canonical<T>(
        &mut self,
        canonical: rustc_type_ir::Canonical<DbInterner<'db>, T>,
    ) -> T
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        self.infer_ctxt.instantiate_canonical(&canonical).0
    }

    pub(crate) fn resolve_completely<T>(&mut self, value: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        let value = self.infer_ctxt.resolve_vars_if_possible(value);

        let mut goals = vec![];

        // FIXME(next-solver): Handle `goals`.

        value.fold_with(&mut resolve_completely::Resolver::new(self, true, &mut goals))
    }

    /// Unify two relatable values (e.g. `Ty`) and register new trait goals that arise from that.
    pub(crate) fn unify<T: ToTrace<'db>>(&mut self, ty1: T, ty2: T) -> bool {
        self.try_unify(ty1, ty2).map(|infer_ok| self.register_infer_ok(infer_ok)).is_ok()
    }

    /// Unify two relatable values (e.g. `Ty`) and return new trait goals arising from it, so the
    /// caller needs to deal with them.
    pub(crate) fn try_unify<T: ToTrace<'db>>(&mut self, t1: T, t2: T) -> InferResult<'db, ()> {
        self.infer_ctxt.at(&ObligationCause::new(), self.trait_env.env).eq(t1, t2)
    }

    pub(crate) fn shallow_resolve(&self, ty: Ty<'db>) -> Ty<'db> {
        self.infer_ctxt.shallow_resolve(ty)
    }

    pub(crate) fn resolve_vars_with_obligations<T>(&mut self, t: T) -> T
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        if !t.has_non_region_infer() {
            return t;
        }

        let t = self.infer_ctxt.resolve_vars_if_possible(t);

        if !t.has_non_region_infer() {
            return t;
        }

        self.select_obligations_where_possible();
        self.infer_ctxt.resolve_vars_if_possible(t)
    }

    /// Create a `GenericArgs` full of infer vars for `def`.
    pub(crate) fn fresh_args_for_item(&self, def: SolverDefId) -> GenericArgs<'db> {
        self.infer_ctxt.fresh_args_for_item(def)
    }

    /// Like `fresh_args_for_item()`, but first uses the args from `first`.
    pub(crate) fn fill_rest_fresh_args(
        &self,
        def_id: SolverDefId,
        first: impl IntoIterator<Item = GenericArg<'db>>,
    ) -> GenericArgs<'db> {
        self.infer_ctxt.fill_rest_fresh_args(def_id, first)
    }

    /// Try to resolve `ty` to a structural type, normalizing aliases.
    ///
    /// In case there is still ambiguity, the returned type may be an inference
    /// variable. This is different from `structurally_resolve_type` which errors
    /// in this case.
    pub(crate) fn try_structurally_resolve_type(&mut self, ty: Ty<'db>) -> Ty<'db> {
        if let TyKind::Alias(..) = ty.kind() {
            // We need to use a separate variable here as otherwise the temporary for
            // `self.fulfillment_cx.borrow_mut()` is alive in the `Err` branch, resulting
            // in a reentrant borrow, causing an ICE.
            let result = self
                .infer_ctxt
                .at(&ObligationCause::misc(), self.trait_env.env)
                .structurally_normalize_ty(ty, &mut self.fulfillment_cx);
            match result {
                Ok(normalized_ty) => normalized_ty,
                Err(_errors) => Ty::new_error(self.interner(), ErrorGuaranteed),
            }
        } else {
            self.resolve_vars_with_obligations(ty)
        }
    }

    pub(crate) fn structurally_resolve_type(&mut self, ty: Ty<'db>) -> Ty<'db> {
        self.try_structurally_resolve_type(ty)
        // FIXME: Err if it still contain infer vars.
    }

    pub(crate) fn snapshot(&mut self) -> InferenceTableSnapshot<'db> {
        let ctxt_snapshot = self.infer_ctxt.start_snapshot();
        let obligations = self.fulfillment_cx.clone();
        InferenceTableSnapshot { ctxt_snapshot, obligations }
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn rollback_to(&mut self, snapshot: InferenceTableSnapshot<'db>) {
        self.infer_ctxt.rollback_to(snapshot.ctxt_snapshot);
        self.fulfillment_cx = snapshot.obligations;
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn run_in_snapshot<T>(
        &mut self,
        f: impl FnOnce(&mut InferenceTable<'db>) -> T,
    ) -> T {
        let snapshot = self.snapshot();
        let result = f(self);
        self.rollback_to(snapshot);
        result
    }

    pub(crate) fn commit_if_ok<T, E>(
        &mut self,
        f: impl FnOnce(&mut InferenceTable<'db>) -> Result<T, E>,
    ) -> Result<T, E> {
        let snapshot = self.snapshot();
        let result = f(self);
        match result {
            Ok(_) => {}
            Err(_) => {
                self.rollback_to(snapshot);
            }
        }
        result
    }

    /// Checks an obligation without registering it. Useful mostly to check
    /// whether a trait *might* be implemented before deciding to 'lock in' the
    /// choice (during e.g. method resolution or deref).
    #[tracing::instrument(level = "debug", skip(self))]
    pub(crate) fn try_obligation(&mut self, predicate: Predicate<'db>) -> NextTraitSolveResult {
        let goal = next_solver::Goal { param_env: self.trait_env.env, predicate };
        let canonicalized = self.canonicalize(goal);

        next_trait_solve_canonical_in_ctxt(&self.infer_ctxt, canonicalized)
    }

    pub(crate) fn register_obligation(&mut self, predicate: Predicate<'db>) {
        let goal = next_solver::Goal { param_env: self.trait_env.env, predicate };
        self.register_obligation_in_env(goal)
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn register_obligation_in_env(
        &mut self,
        goal: next_solver::Goal<'db, next_solver::Predicate<'db>>,
    ) {
        let result = next_trait_solve_in_ctxt(&self.infer_ctxt, goal);
        tracing::debug!(?result);
        match result {
            Ok((_, Certainty::Yes)) => {}
            Err(rustc_type_ir::solve::NoSolution) => {}
            Ok((_, Certainty::Maybe { .. })) => {
                self.fulfillment_cx.register_predicate_obligation(
                    &self.infer_ctxt,
                    Obligation::new(
                        self.interner(),
                        ObligationCause::new(),
                        goal.param_env,
                        goal.predicate,
                    ),
                );
            }
        }
    }

    pub(crate) fn register_infer_ok<T>(&mut self, infer_ok: InferOk<'db, T>) -> T {
        let InferOk { value, obligations } = infer_ok;
        self.register_predicates(obligations);
        value
    }

    pub(crate) fn select_obligations_where_possible(&mut self) {
        self.fulfillment_cx.try_evaluate_obligations(&self.infer_ctxt);
    }

    pub(super) fn register_predicate(&mut self, obligation: PredicateObligation<'db>) {
        if obligation.has_escaping_bound_vars() {
            panic!("escaping bound vars in predicate {:?}", obligation);
        }

        self.fulfillment_cx.register_predicate_obligation(&self.infer_ctxt, obligation);
    }

    pub(super) fn register_predicates<I>(&mut self, obligations: I)
    where
        I: IntoIterator<Item = PredicateObligation<'db>>,
    {
        obligations.into_iter().for_each(|obligation| {
            self.register_predicate(obligation);
        });
    }

    pub(crate) fn callable_sig(
        &mut self,
        ty: Ty<'db>,
        num_args: usize,
    ) -> Option<(Option<FnTrait>, Vec<Ty<'db>>, Ty<'db>)> {
        match ty.callable_sig(self.interner()) {
            Some(sig) => {
                let sig = sig.skip_binder();
                Some((None, sig.inputs_and_output.inputs().to_vec(), sig.output()))
            }
            None => {
                let (f, args_ty, return_ty) = self.callable_sig_from_fn_trait(ty, num_args)?;
                Some((Some(f), args_ty, return_ty))
            }
        }
    }

    fn callable_sig_from_fn_trait(
        &mut self,
        ty: Ty<'db>,
        num_args: usize,
    ) -> Option<(FnTrait, Vec<Ty<'db>>, Ty<'db>)> {
        for (fn_trait_name, output_assoc_name, subtraits) in [
            (FnTrait::FnOnce, sym::Output, &[FnTrait::Fn, FnTrait::FnMut][..]),
            (FnTrait::AsyncFnMut, sym::CallRefFuture, &[FnTrait::AsyncFn]),
            (FnTrait::AsyncFnOnce, sym::CallOnceFuture, &[]),
        ] {
            let krate = self.trait_env.krate;
            let fn_trait = fn_trait_name.get_id(self.db, krate)?;
            let trait_data = fn_trait.trait_items(self.db);
            let output_assoc_type =
                trait_data.associated_type_by_name(&Name::new_symbol_root(output_assoc_name))?;

            let mut arg_tys = Vec::with_capacity(num_args);
            let arg_ty = Ty::new_tup_from_iter(
                self.interner(),
                std::iter::repeat_with(|| {
                    let ty = self.next_ty_var();
                    arg_tys.push(ty);
                    ty
                })
                .take(num_args),
            );
            let args = [ty, arg_ty];
            let trait_ref = TraitRef::new(self.interner(), fn_trait.into(), args);

            let proj_args = self
                .infer_ctxt
                .fill_rest_fresh_args(output_assoc_type.into(), args.into_iter().map(Into::into));
            let projection = Ty::new_alias(
                self.interner(),
                rustc_type_ir::AliasTyKind::Projection,
                AliasTy::new(self.interner(), output_assoc_type.into(), proj_args),
            );

            let pred = Predicate::upcast_from(trait_ref, self.interner());
            if !self.try_obligation(pred).no_solution() {
                self.register_obligation(pred);
                let return_ty = self.normalize_alias_ty(projection);
                for &fn_x in subtraits {
                    let fn_x_trait = fn_x.get_id(self.db, krate)?;
                    let trait_ref = TraitRef::new(self.interner(), fn_x_trait.into(), args);
                    let pred = Predicate::upcast_from(trait_ref, self.interner());
                    if !self.try_obligation(pred).no_solution() {
                        return Some((fn_x, arg_tys, return_ty));
                    }
                }
                return Some((fn_trait_name, arg_tys, return_ty));
            }
        }
        None
    }

    pub(super) fn insert_type_vars<T>(&mut self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        self.infer_ctxt.insert_type_vars(ty)
    }

    /// Replaces `Ty::Error` by a new type var, so we can maybe still infer it.
    pub(super) fn insert_type_vars_shallow(&mut self, ty: Ty<'db>) -> Ty<'db> {
        if ty.is_ty_error() { self.next_ty_var() } else { ty }
    }

    /// Whenever you lower a user-written type, you should call this.
    pub(crate) fn process_user_written_ty<T>(&mut self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        self.process_remote_user_written_ty(ty)
        // FIXME: Register a well-formed obligation.
    }

    /// The difference of this method from `process_user_written_ty()` is that this method doesn't register a well-formed obligation,
    /// while `process_user_written_ty()` should (but doesn't currently).
    pub(crate) fn process_remote_user_written_ty<T>(&mut self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        let ty = self.insert_type_vars(ty);
        // See https://github.com/rust-lang/rust/blob/cdb45c87e2cd43495379f7e867e3cc15dcee9f93/compiler/rustc_hir_typeck/src/fn_ctxt/mod.rs#L487-L495:
        // Even though the new solver only lazily normalizes usually, here we eagerly normalize so that not everything needs
        // to normalize before inspecting the `TyKind`.
        // FIXME(next-solver): We should not deeply normalize here, only shallowly.
        self.normalize_associated_types_in(ty)
    }

    /// Replaces ConstScalar::Unknown by a new type var, so we can maybe still infer it.
    pub(super) fn insert_const_vars_shallow(&mut self, c: Const<'db>) -> Const<'db> {
        if c.is_ct_error() { self.next_const_var() } else { c }
    }

    /// Check if given type is `Sized` or not
    pub(crate) fn is_sized(&mut self, ty: Ty<'db>) -> bool {
        fn short_circuit_trivial_tys(ty: Ty<'_>) -> Option<bool> {
            match ty.kind() {
                TyKind::Bool
                | TyKind::Char
                | TyKind::Int(_)
                | TyKind::Uint(_)
                | TyKind::Float(_)
                | TyKind::Ref(..)
                | TyKind::RawPtr(..)
                | TyKind::Never
                | TyKind::FnDef(..)
                | TyKind::Array(..)
                | TyKind::FnPtr(..) => Some(true),
                TyKind::Slice(..) | TyKind::Str | TyKind::Dynamic(..) => Some(false),
                _ => None,
            }
        }

        let mut ty = ty;
        ty = self.eagerly_normalize_and_resolve_shallow_in(ty);
        if let Some(sized) = short_circuit_trivial_tys(ty) {
            return sized;
        }

        {
            let mut structs = SmallVec::<[_; 8]>::new();
            // Must use a loop here and not recursion because otherwise users will conduct completely
            // artificial examples of structs that have themselves as the tail field and complain r-a crashes.
            while let Some((AdtId::StructId(id), subst)) = ty.as_adt() {
                let struct_data = id.fields(self.db);
                if let Some((last_field, _)) = struct_data.fields().iter().next_back() {
                    let last_field_ty = self.db.field_types(id.into())[last_field]
                        .instantiate(self.interner(), subst);
                    if structs.contains(&ty) {
                        // A struct recursively contains itself as a tail field somewhere.
                        return true; // Don't overload the users with too many errors.
                    }
                    structs.push(ty);
                    // Structs can have DST as its last field and such cases are not handled
                    // as unsized by the chalk, so we do this manually.
                    ty = last_field_ty;
                    ty = self.eagerly_normalize_and_resolve_shallow_in(ty);
                    if let Some(sized) = short_circuit_trivial_tys(ty) {
                        return sized;
                    }
                } else {
                    break;
                };
            }
        }

        let Some(sized) = LangItem::Sized.resolve_trait(self.db, self.trait_env.krate) else {
            return false;
        };
        let sized_pred = Predicate::upcast_from(
            TraitRef::new(self.interner(), sized.into(), [ty]),
            self.interner(),
        );
        self.try_obligation(sized_pred).certain()
    }
}

impl fmt::Debug for InferenceTable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceTable")
            .field("name", &self.infer_ctxt.inner.borrow().type_variable_storage)
            .field("fulfillment_cx", &self.fulfillment_cx)
            .finish()
    }
}

mod resolve_completely {
    use rustc_type_ir::{DebruijnIndex, Flags, TypeFolder, TypeSuperFoldable};

    use crate::{
        infer::unify::InferenceTable,
        next_solver::{
            Const, DbInterner, Goal, Predicate, Region, Term, Ty,
            infer::{resolve::ReplaceInferWithError, traits::ObligationCause},
            normalize::deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals,
        },
    };

    pub(super) struct Resolver<'a, 'db> {
        ctx: &'a mut InferenceTable<'db>,
        /// Whether we should normalize, disabled when resolving predicates.
        should_normalize: bool,
        nested_goals: &'a mut Vec<Goal<'db, Predicate<'db>>>,
    }

    impl<'a, 'db> Resolver<'a, 'db> {
        pub(super) fn new(
            ctx: &'a mut InferenceTable<'db>,
            should_normalize: bool,
            nested_goals: &'a mut Vec<Goal<'db, Predicate<'db>>>,
        ) -> Resolver<'a, 'db> {
            Resolver { ctx, nested_goals, should_normalize }
        }

        fn handle_term<T>(
            &mut self,
            value: T,
            outer_exclusive_binder: impl FnOnce(T) -> DebruijnIndex,
        ) -> T
        where
            T: Into<Term<'db>> + TypeSuperFoldable<DbInterner<'db>> + Copy,
        {
            let value = if self.should_normalize {
                let cause = ObligationCause::new();
                let at = self.ctx.infer_ctxt.at(&cause, self.ctx.trait_env.env);
                let universes = vec![None; outer_exclusive_binder(value).as_usize()];
                match deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
                    at, value, universes,
                ) {
                    Ok((value, goals)) => {
                        self.nested_goals.extend(goals);
                        value
                    }
                    Err(_errors) => {
                        // FIXME: Report the error.
                        value
                    }
                }
            } else {
                value
            };

            value.fold_with(&mut ReplaceInferWithError::new(self.ctx.interner()))
        }
    }

    impl<'cx, 'db> TypeFolder<DbInterner<'db>> for Resolver<'cx, 'db> {
        fn cx(&self) -> DbInterner<'db> {
            self.ctx.interner()
        }

        fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
            if r.is_var() { Region::error(self.ctx.interner()) } else { r }
        }

        fn fold_ty(&mut self, ty: Ty<'db>) -> Ty<'db> {
            self.handle_term(ty, |it| it.outer_exclusive_binder())
        }

        fn fold_const(&mut self, ct: Const<'db>) -> Const<'db> {
            self.handle_term(ct, |it| it.outer_exclusive_binder())
        }

        fn fold_predicate(&mut self, predicate: Predicate<'db>) -> Predicate<'db> {
            assert!(
                !self.should_normalize,
                "normalizing predicates in writeback is not generally sound"
            );
            predicate.super_fold_with(self)
        }
    }
}

//! Unification and canonicalization logic.

use std::fmt;

use base_db::Crate;
use hir_def::{ExpressionStoreOwnerId, GenericParamId, TraitId};
use rustc_hash::FxHashSet;
use rustc_type_ir::{
    TyVid, TypeFoldable, TypeVisitableExt,
    inherent::{Const as _, GenericArg as _, IntoKind, Ty as _},
    solve::Certainty,
};
use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::{
    InferenceDiagnostic, Span,
    db::HirDatabase,
    next_solver::{
        Canonical, ClauseKind, Const, ConstKind, DbInterner, ErrorGuaranteed, GenericArg,
        GenericArgs, ParamEnv, Predicate, PredicateKind, Region, SolverDefId, Term, TraitRef, Ty,
        TyKind, TypingMode,
        fulfill::{FulfillmentCtxt, NextSolverError},
        infer::{
            DbInternerInferExt, InferCtxt, InferOk,
            at::At,
            snapshot::CombinedSnapshot,
            traits::{Obligation, ObligationCause, PredicateObligation},
        },
        inspect::{InspectConfig, InspectGoal, ProofTreeVisitor},
        obligation_ctxt::ObligationCtxt,
    },
    solver_errors::SolverDiagnostic,
    traits::ParamEnvAndCrate,
};

struct NestedObligationsForSelfTy<'a, 'db> {
    ctx: &'a InferenceTable<'db>,
    self_ty: TyVid,
    root_cause: &'a ObligationCause,
    obligations_for_self_ty: &'a mut SmallVec<[Obligation<'db, Predicate<'db>>; 4]>,
}

impl<'a, 'db> ProofTreeVisitor<'db> for NestedObligationsForSelfTy<'a, 'db> {
    type Result = ();

    fn span(&self) -> Span {
        self.root_cause.span()
    }

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
        if self.ctx.predicate_has_self_ty(goal.predicate, self.self_ty) {
            self.obligations_for_self_ty.push(Obligation::new(
                db,
                *self.root_cause,
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
    env: ParamEnvAndCrate<'db>,
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
    env: ParamEnvAndCrate<'db>,
    tys: &Canonical<'db, (Ty<'db>, Ty<'db>)>,
) -> bool {
    could_unify_impl(db, env, tys, |ctxt| ctxt.evaluate_obligations_error_on_ambiguity())
}

fn could_unify_impl<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    tys: &Canonical<'db, (Ty<'db>, Ty<'db>)>,
    select: for<'a> fn(&mut ObligationCtxt<'a, 'db>) -> Vec<NextSolverError<'db>>,
) -> bool {
    let interner = DbInterner::new_with(db, env.krate);
    let infcx = interner.infer_ctxt().build(TypingMode::PostAnalysis);
    let cause = ObligationCause::dummy();
    let at = infcx.at(&cause, env.param_env);
    let ((ty1_with_vars, ty2_with_vars), _) = infcx.instantiate_canonical(Span::Dummy, tys);
    let mut ctxt = ObligationCtxt::new(&infcx);
    let can_unify = at
        .eq(ty1_with_vars, ty2_with_vars)
        .map(|infer_ok| ctxt.register_infer_ok_obligations(infer_ok))
        .is_ok();
    can_unify && select(&mut ctxt).is_empty()
}

pub(crate) struct InferenceTable<'db> {
    pub(crate) db: &'db dyn HirDatabase,
    pub(crate) param_env: ParamEnv<'db>,
    pub(crate) infer_ctxt: InferCtxt<'db>,
    pub(super) fulfillment_cx: FulfillmentCtxt<'db>,
    pub(super) diverging_type_vars: FxHashSet<Ty<'db>>,
    pub(super) trait_errors: Vec<NextSolverError<'db>>,
}

impl<'db> InferenceTable<'db> {
    /// Inside hir-ty you should use this for inference only, and always pass `owner`.
    /// Outside it, always pass `owner = None`.
    pub(crate) fn new(
        db: &'db dyn HirDatabase,
        trait_env: ParamEnv<'db>,
        krate: Crate,
        owner: ExpressionStoreOwnerId,
    ) -> Self {
        let interner = DbInterner::new_with(db, krate);
        let typing_mode = TypingMode::typeck_for_body(interner, owner.into());
        let infer_ctxt = interner.infer_ctxt().build(typing_mode);
        InferenceTable {
            db,
            param_env: trait_env,
            fulfillment_cx: FulfillmentCtxt::new(&infer_ctxt),
            infer_ctxt,
            diverging_type_vars: FxHashSet::default(),
            trait_errors: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn interner(&self) -> DbInterner<'db> {
        self.infer_ctxt.interner
    }

    pub(crate) fn type_is_copy_modulo_regions(&self, ty: Ty<'db>) -> bool {
        self.infer_ctxt.type_is_copy_modulo_regions(self.param_env, ty)
    }

    pub(crate) fn type_is_sized_modulo_regions(&self, ty: Ty<'db>) -> bool {
        self.infer_ctxt.type_is_sized_modulo_regions(self.param_env, ty)
    }

    pub(crate) fn type_is_use_cloned_modulo_regions(&self, ty: Ty<'db>) -> bool {
        self.infer_ctxt.type_is_use_cloned_modulo_regions(self.param_env, ty)
    }

    pub(crate) fn type_var_is_sized(&self, self_ty: TyVid) -> bool {
        let Some(sized_did) = self.interner().lang_items().Sized else {
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

    pub(crate) fn next_ty_var(&self, span: Span) -> Ty<'db> {
        self.infer_ctxt.next_ty_var(span)
    }

    pub(crate) fn next_const_var(&self, span: Span) -> Const<'db> {
        self.infer_ctxt.next_const_var(span)
    }

    pub(crate) fn next_int_var(&self) -> Ty<'db> {
        self.infer_ctxt.next_int_var()
    }

    pub(crate) fn next_float_var(&self) -> Ty<'db> {
        self.infer_ctxt.next_float_var()
    }

    pub(crate) fn new_maybe_never_var(&mut self, span: Span) -> Ty<'db> {
        let var = self.next_ty_var(span);
        self.set_diverging(var);
        var
    }

    pub(crate) fn next_region_var(&self, span: Span) -> Region<'db> {
        self.infer_ctxt.next_region_var(span)
    }

    pub(crate) fn var_for_def(&self, id: GenericParamId, span: Span) -> GenericArg<'db> {
        self.infer_ctxt.var_for_def(id, span)
    }

    pub(crate) fn at<'a>(&'a self, cause: &'a ObligationCause) -> At<'a, 'db> {
        self.infer_ctxt.at(cause, self.param_env)
    }

    pub(crate) fn shallow_resolve(&self, ty: Ty<'db>) -> Ty<'db> {
        self.infer_ctxt.shallow_resolve(ty)
    }

    pub(crate) fn resolve_vars_if_possible<T: TypeFoldable<DbInterner<'db>>>(&self, t: T) -> T {
        self.infer_ctxt.resolve_vars_if_possible(t)
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
    pub(crate) fn fresh_args_for_item(&self, span: Span, def: SolverDefId) -> GenericArgs<'db> {
        self.infer_ctxt.fresh_args_for_item(span, def)
    }

    /// Try to resolve `ty` to a structural type, normalizing aliases.
    ///
    /// In case there is still ambiguity, the returned type may be an inference
    /// variable. This is different from `structurally_resolve_type` which errors
    /// in this case.
    pub(crate) fn try_structurally_resolve_type(&mut self, span: Span, ty: Ty<'db>) -> Ty<'db> {
        if let TyKind::Alias(..) = ty.kind() {
            let result = self
                .infer_ctxt
                .at(&ObligationCause::new(span), self.param_env)
                .structurally_normalize_ty(ty, &mut self.fulfillment_cx);
            match result {
                Ok(normalized_ty) => normalized_ty,
                Err(errors) => {
                    self.trait_errors.extend(errors);
                    Ty::new_error(self.interner(), ErrorGuaranteed)
                }
            }
        } else {
            self.resolve_vars_with_obligations(ty)
        }
    }

    pub(crate) fn try_structurally_resolve_const(
        &mut self,
        sp: Span,
        ct: Const<'db>,
    ) -> Const<'db> {
        let ct = self.resolve_vars_with_obligations(ct);

        if let ConstKind::Unevaluated(..) = ct.kind() {
            let result = self
                .infer_ctxt
                .at(&ObligationCause::new(sp), self.param_env)
                .structurally_normalize_const(ct, &mut self.fulfillment_cx);
            match result {
                Ok(normalized_ct) => normalized_ct,
                Err(errors) => {
                    self.trait_errors.extend(errors);
                    Const::new_error(self.interner(), ErrorGuaranteed)
                }
            }
        } else {
            ct
        }
    }

    pub(crate) fn snapshot(&mut self) -> CombinedSnapshot {
        self.infer_ctxt.start_snapshot()
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn rollback_to(&mut self, snapshot: CombinedSnapshot) {
        self.infer_ctxt.rollback_to(snapshot);
    }

    pub(crate) fn commit_if_ok<T, E>(
        &mut self,
        f: impl FnOnce(&mut InferenceTable<'db>) -> Result<T, E>,
    ) -> Result<T, E> {
        let snapshot = self.snapshot();
        let result = f(self);
        match result {
            Ok(_) => self.infer_ctxt.commit_from(snapshot),
            Err(_) => self.rollback_to(snapshot),
        }
        result
    }

    pub(crate) fn register_bound(&mut self, ty: Ty<'db>, def_id: TraitId, cause: ObligationCause) {
        if !ty.references_non_lt_error() {
            let trait_ref = TraitRef::new(self.interner(), def_id.into(), [ty]);
            self.register_predicate(Obligation::new(
                self.interner(),
                cause,
                self.param_env,
                trait_ref,
            ));
        }
    }

    pub(crate) fn register_infer_ok<T>(&mut self, infer_ok: InferOk<'db, T>) -> T {
        let InferOk { value, obligations } = infer_ok;
        self.register_predicates(obligations);
        value
    }

    pub(crate) fn select_obligations_where_possible(&mut self) {
        let errors = self.fulfillment_cx.try_evaluate_obligations(&self.infer_ctxt);
        self.trait_errors.extend(errors);
    }

    pub(super) fn register_predicate(&mut self, obligation: PredicateObligation<'db>) {
        if obligation.has_escaping_bound_vars() {
            panic!("escaping bound vars in predicate {:?}", obligation);
        }

        self.fulfillment_cx.register_predicate_obligation(&self.infer_ctxt, obligation);
    }

    pub(crate) fn register_predicates<I>(&mut self, obligations: I)
    where
        I: IntoIterator<Item = PredicateObligation<'db>>,
    {
        self.fulfillment_cx.register_predicate_obligations(&self.infer_ctxt, obligations);
    }

    /// checking later, during regionck, that `arg` is well-formed.
    pub(crate) fn register_wf_obligation(&mut self, term: Term<'db>, cause: ObligationCause) {
        self.register_predicate(Obligation::new(
            self.interner(),
            cause,
            self.param_env,
            ClauseKind::WellFormed(term),
        ));
    }

    /// Registers obligations that all `args` are well-formed.
    pub(crate) fn add_wf_bounds(&mut self, span: Span, args: GenericArgs<'db>) {
        for term in args.iter().filter_map(|it| it.as_term()) {
            self.register_wf_obligation(term, ObligationCause::new(span));
        }
    }

    pub(super) fn insert_type_vars<T>(&mut self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        self.infer_ctxt.insert_type_vars(ty)
    }

    /// Whenever you lower a user-written type, you should call this.
    pub(crate) fn process_user_written_ty(&mut self, ty: Ty<'db>) -> Ty<'db> {
        self.process_remote_user_written_ty(ty)
    }

    /// The difference of this method from `process_user_written_ty()` is that this method doesn't register a well-formed obligation,
    /// while `process_user_written_ty()` should (but doesn't currently).
    pub(crate) fn process_remote_user_written_ty(&mut self, ty: Ty<'db>) -> Ty<'db> {
        let ty = self.insert_type_vars(ty);
        // See https://github.com/rust-lang/rust/blob/cdb45c87e2cd43495379f7e867e3cc15dcee9f93/compiler/rustc_hir_typeck/src/fn_ctxt/mod.rs#L487-L495:
        // Even though the new solver only lazily normalizes usually, here we eagerly normalize so that not everything needs
        // to normalize before inspecting the `TyKind`.
        self.try_structurally_resolve_type(Span::Dummy, ty)
    }

    fn emit_trait_errors(&mut self, diagnostics: &mut ThinVec<InferenceDiagnostic>) {
        diagnostics.extend(std::mem::take(&mut self.trait_errors).into_iter().filter_map(
            |error| {
                let error = error.into_fulfillment_error(&self.infer_ctxt);
                SolverDiagnostic::from_fulfillment_error(&error)
                    .map(InferenceDiagnostic::SolverDiagnostic)
            },
        ));
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

pub(super) mod resolve_completely {
    use rustc_hash::FxHashSet;
    use rustc_type_ir::{
        DebruijnIndex, Flags, InferConst, InferTy, TypeFlags, TypeFoldable, TypeFolder,
        TypeSuperFoldable, TypeVisitableExt, inherent::IntoKind,
    };
    use stdx::never;
    use thin_vec::ThinVec;

    use crate::{
        InferenceDiagnostic, Span,
        infer::unify::InferenceTable,
        next_solver::{
            Const, ConstKind, DbInterner, DefaultAny, GenericArg, Goal, Predicate, Region, Term,
            TermKind, Ty, TyKind,
            infer::{resolve::ReplaceInferWithError, traits::ObligationCause},
            normalize::deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals,
        },
    };

    pub(crate) struct WriteBackCtxt<'db> {
        table: InferenceTable<'db>,
        diagnostics: ThinVec<InferenceDiagnostic>,
        has_errors: bool,
        spans_emitted_type_must_be_known_for: FxHashSet<Span>,
        types: &'db DefaultAny<'db>,
    }

    impl<'db> WriteBackCtxt<'db> {
        pub(crate) fn new(
            table: InferenceTable<'db>,
            diagnostics: ThinVec<InferenceDiagnostic>,
            vars_emitted_type_must_be_known_for: FxHashSet<Term<'db>>,
        ) -> Self {
            let spans_emitted_type_must_be_known_for = vars_emitted_type_must_be_known_for
                .into_iter()
                .filter_map(|term| match term.kind() {
                    TermKind::Ty(ty) => match ty.kind() {
                        TyKind::Infer(InferTy::TyVar(vid)) => {
                            Some(table.infer_ctxt.type_var_span(vid))
                        }
                        _ => None,
                    },
                    TermKind::Const(ct) => match ct.kind() {
                        ConstKind::Infer(InferConst::Var(vid)) => {
                            table.infer_ctxt.const_var_span(vid)
                        }
                        _ => None,
                    },
                })
                .collect();

            Self {
                types: table.interner().default_types(),
                table,
                diagnostics,
                has_errors: false,
                spans_emitted_type_must_be_known_for,
            }
        }

        pub(crate) fn resolve_completely<T>(&mut self, value_ref: &mut T)
        where
            T: TypeFoldable<DbInterner<'db>>,
        {
            self.resolve_completely_with_default(value_ref, value_ref.clone());
        }

        pub(crate) fn resolve_completely_with_default<T>(&mut self, value_ref: &mut T, default: T)
        where
            T: TypeFoldable<DbInterner<'db>>,
        {
            let value = std::mem::replace(value_ref, default);

            let value = self.table.resolve_vars_if_possible(value);

            let mut goals = vec![];

            // FIXME(next-solver): Handle `goals`.

            *value_ref = value.fold_with(&mut Resolver::new(self, true, &mut goals));
        }

        pub(crate) fn resolve_diagnostics(mut self) -> (ThinVec<InferenceDiagnostic>, bool) {
            let has_errors = self.has_errors;

            self.table.emit_trait_errors(&mut self.diagnostics);

            // Ignore diagnostics made from resolving diagnostics.
            let mut diagnostics = std::mem::take(&mut self.diagnostics);
            diagnostics.retain_mut(|diagnostic| {
                self.resolve_completely(diagnostic);

                if let InferenceDiagnostic::ExpectedFunction { found: ty, .. }
                | InferenceDiagnostic::ExpectedArrayOrSlicePat { found: ty, .. }
                | InferenceDiagnostic::UnresolvedField { receiver: ty, .. }
                | InferenceDiagnostic::UnresolvedMethodCall { receiver: ty, .. } = diagnostic
                    && ty.as_ref().references_non_lt_error()
                {
                    false
                } else {
                    true
                }
            });
            diagnostics.shrink_to_fit();

            (diagnostics, has_errors)
        }
    }

    struct DiagnoseInferVars<'a, 'db> {
        ctx: &'a mut WriteBackCtxt<'db>,
        top_term: Term<'db>,
    }

    impl<'db> DiagnoseInferVars<'_, 'db> {
        const TYPE_FLAGS: TypeFlags = TypeFlags::HAS_INFER.union(TypeFlags::HAS_NON_REGION_ERROR);

        fn err_on_span(&mut self, span: Span) {
            if !self.ctx.spans_emitted_type_must_be_known_for.insert(span) {
                // Suppress duplicate diagnostics.
                return;
            }

            if span.is_dummy() {
                return;
            }

            // We have to be careful not to insert infer vars here, as we won't resolve this new diagnostic.
            let top_term = self.top_term.fold_with(&mut ReplaceInferWithError::new(self.cx()));
            self.ctx.diagnostics.push(InferenceDiagnostic::TypeMustBeKnown {
                at_point: span,
                top_term: Some(GenericArg::from(top_term).store()),
            });
        }
    }

    impl<'db> TypeFolder<DbInterner<'db>> for DiagnoseInferVars<'_, 'db> {
        fn cx(&self) -> DbInterner<'db> {
            self.ctx.table.interner()
        }

        fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
            if !t.has_type_flags(Self::TYPE_FLAGS) {
                return t;
            }

            match t.kind() {
                TyKind::Error(_) => {
                    self.ctx.has_errors = true;
                    t
                }
                TyKind::Infer(infer_ty) => match infer_ty {
                    InferTy::TyVar(vid) => {
                        self.err_on_span(self.ctx.table.infer_ctxt.type_var_span(vid));
                        self.ctx.has_errors = true;
                        self.ctx.types.types.error
                    }
                    InferTy::IntVar(_) => {
                        never!("fallback should have resolved all int vars");
                        self.ctx.types.types.i32
                    }
                    InferTy::FloatVar(_) => {
                        never!("fallback should have resolved all float vars");
                        self.ctx.types.types.f64
                    }
                    InferTy::FreshTy(_) | InferTy::FreshIntTy(_) | InferTy::FreshFloatTy(_) => {
                        never!("should not have fresh infer vars outside of caching");
                        self.ctx.has_errors = true;
                        self.ctx.types.types.error
                    }
                },
                _ => t.super_fold_with(self),
            }
        }

        fn fold_const(&mut self, c: Const<'db>) -> Const<'db> {
            if !c.has_type_flags(Self::TYPE_FLAGS) {
                return c;
            }

            match c.kind() {
                ConstKind::Error(_) => {
                    self.ctx.has_errors = true;
                    c
                }
                ConstKind::Infer(infer_ct) => match infer_ct {
                    InferConst::Var(vid) => {
                        if let Some(span) = self.ctx.table.infer_ctxt.const_var_span(vid) {
                            self.err_on_span(span);
                        }
                        self.ctx.has_errors = true;
                        self.ctx.types.consts.error
                    }
                    InferConst::Fresh(_) => {
                        never!("should not have fresh infer vars outside of caching");
                        self.ctx.has_errors = true;
                        self.ctx.types.consts.error
                    }
                },
                _ => c.super_fold_with(self),
            }
        }

        fn fold_predicate(&mut self, p: Predicate<'db>) -> Predicate<'db> {
            if !p.has_type_flags(Self::TYPE_FLAGS) {
                return p;
            }
            p.super_fold_with(self)
        }

        fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
            if r.is_var() {
                // For now, we don't error on regions.
                self.ctx.types.regions.error
            } else {
                r
            }
        }
    }

    pub(super) struct Resolver<'a, 'db> {
        ctx: &'a mut WriteBackCtxt<'db>,
        /// Whether we should normalize, disabled when resolving predicates.
        should_normalize: bool,
        nested_goals: &'a mut Vec<Goal<'db, Predicate<'db>>>,
    }

    impl<'a, 'db> Resolver<'a, 'db> {
        pub(super) fn new(
            ctx: &'a mut WriteBackCtxt<'db>,
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
                // FIXME: This should not use a dummy span.
                let cause = ObligationCause::new(Span::Dummy);
                let at = self.ctx.table.at(&cause);
                let universes = vec![None; outer_exclusive_binder(value).as_usize()];
                match deeply_normalize_with_skipped_universes_and_ambiguous_coroutine_goals(
                    at, value, universes,
                ) {
                    Ok((value, goals)) => {
                        self.nested_goals.extend(goals);
                        value
                    }
                    Err(errors) => {
                        self.ctx.table.trait_errors.extend(errors);
                        value
                    }
                }
            } else {
                value
            };

            value.fold_with(&mut DiagnoseInferVars { ctx: self.ctx, top_term: value.into() })
        }
    }

    impl<'db> TypeFolder<DbInterner<'db>> for Resolver<'_, 'db> {
        fn cx(&self) -> DbInterner<'db> {
            self.ctx.table.interner()
        }

        fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
            if r.is_var() { self.ctx.types.regions.error } else { r }
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

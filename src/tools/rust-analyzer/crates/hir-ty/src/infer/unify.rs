//! Unification and canonicalization logic.

use std::fmt;

use chalk_ir::{
    CanonicalVarKind, TyVariableKind, cast::Cast, fold::TypeFoldable, interner::HasInterner,
};
use either::Either;
use hir_def::{AdtId, lang_item::LangItem};
use hir_expand::name::Name;
use intern::sym;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::{
    TyVid, TypeVisitableExt, UpcastFrom,
    inherent::{IntoKind, Span, Term as _, Ty as _},
    relate::{Relate, solver_relating::RelateExt},
    solve::{Certainty, GoalSource},
};
use smallvec::SmallVec;
use triomphe::Arc;

use super::{InferResult, InferenceContext, TypeError};
use crate::{
    AliasTy, BoundVar, Canonical, Const, ConstValue, DebruijnIndex, GenericArg, GenericArgData,
    InferenceVar, Interner, Lifetime, OpaqueTyId, ProjectionTy, Substitution, TraitEnvironment, Ty,
    TyExt, TyKind, VariableKind,
    consteval::unknown_const,
    db::HirDatabase,
    fold_generic_args, fold_tys_and_consts,
    next_solver::{
        self, ClauseKind, DbInterner, ErrorGuaranteed, Predicate, PredicateKind, SolverDefIds,
        Term, TraitRef,
        fulfill::FulfillmentCtxt,
        infer::{
            DbInternerInferExt, InferCtxt, InferOk,
            snapshot::CombinedSnapshot,
            traits::{Obligation, ObligationCause},
        },
        inspect::{InspectConfig, InspectGoal, ProofTreeVisitor},
        mapping::{ChalkToNextSolver, NextSolverToChalk},
    },
    traits::{
        FnTrait, NextTraitSolveResult, next_trait_solve_canonical_in_ctxt, next_trait_solve_in_ctxt,
    },
};

impl<'db> InferenceContext<'db> {
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

        let db = self.ctx.interner;
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
pub fn could_unify(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment<'_>>,
    tys: &Canonical<(Ty, Ty)>,
) -> bool {
    unify(db, env, tys).is_some()
}

/// Check if types unify eagerly making sure there are no unresolved goals.
///
/// This means that placeholder types are not considered to unify if there are any bounds set on
/// them. For example `Option<T>` and `Option<U>` do not unify as we cannot show that `T = U`
pub fn could_unify_deeply(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment<'_>>,
    tys: &Canonical<(Ty, Ty)>,
) -> bool {
    let mut table = InferenceTable::new(db, env);
    let vars = make_substitutions(tys, &mut table);
    let ty1_with_vars = vars.apply(tys.value.0.clone(), Interner);
    let ty2_with_vars = vars.apply(tys.value.1.clone(), Interner);
    let ty1_with_vars = table.normalize_associated_types_in(ty1_with_vars);
    let ty2_with_vars = table.normalize_associated_types_in(ty2_with_vars);
    table.select_obligations_where_possible();
    let ty1_with_vars = table.resolve_completely(ty1_with_vars);
    let ty2_with_vars = table.resolve_completely(ty2_with_vars);
    table.unify_deeply(&ty1_with_vars, &ty2_with_vars)
}

pub(crate) fn unify(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment<'_>>,
    tys: &Canonical<(Ty, Ty)>,
) -> Option<Substitution> {
    let mut table = InferenceTable::new(db, env);
    let vars = make_substitutions(tys, &mut table);
    let ty1_with_vars = vars.apply(tys.value.0.clone(), Interner);
    let ty2_with_vars = vars.apply(tys.value.1.clone(), Interner);
    if !table.unify(&ty1_with_vars, &ty2_with_vars) {
        return None;
    }
    // default any type vars that weren't unified back to their original bound vars
    // (kind of hacky)
    let find_var = |iv| {
        vars.iter(Interner).position(|v| match v.data(Interner) {
            GenericArgData::Ty(ty) => ty.inference_var(Interner),
            GenericArgData::Lifetime(lt) => lt.inference_var(Interner),
            GenericArgData::Const(c) => c.inference_var(Interner),
        } == Some(iv))
    };
    let fallback = |iv, kind, binder| match kind {
        chalk_ir::VariableKind::Ty(_ty_kind) => find_var(iv).map_or_else(
            || TyKind::Error.intern(Interner).cast(Interner),
            |i| BoundVar::new(binder, i).to_ty(Interner).cast(Interner),
        ),
        chalk_ir::VariableKind::Lifetime => find_var(iv).map_or_else(
            || crate::error_lifetime().cast(Interner),
            |i| BoundVar::new(binder, i).to_lifetime(Interner).cast(Interner),
        ),
        chalk_ir::VariableKind::Const(ty) => find_var(iv).map_or_else(
            || crate::unknown_const(ty.clone()).cast(Interner),
            |i| BoundVar::new(binder, i).to_const(Interner, ty.clone()).cast(Interner),
        ),
    };
    Some(Substitution::from_iter(
        Interner,
        vars.iter(Interner).map(|v| table.resolve_with_fallback(v.clone(), &fallback)),
    ))
}

fn make_substitutions(
    tys: &chalk_ir::Canonical<(chalk_ir::Ty<Interner>, chalk_ir::Ty<Interner>)>,
    table: &mut InferenceTable<'_>,
) -> chalk_ir::Substitution<Interner> {
    Substitution::from_iter(
        Interner,
        tys.binders.iter(Interner).map(|it| match &it.kind {
            chalk_ir::VariableKind::Ty(_) => table.new_type_var().cast(Interner),
            // FIXME: maybe wrong?
            chalk_ir::VariableKind::Lifetime => table.new_type_var().cast(Interner),
            chalk_ir::VariableKind::Const(ty) => table.new_const_var(ty.clone()).cast(Interner),
        }),
    )
}

bitflags::bitflags! {
    #[derive(Default, Clone, Copy)]
    pub(crate) struct TypeVariableFlags: u8 {
        const DIVERGING = 1 << 0;
        const INTEGER = 1 << 1;
        const FLOAT = 1 << 2;
    }
}

#[derive(Clone)]
pub(crate) struct InferenceTable<'db> {
    pub(crate) db: &'db dyn HirDatabase,
    pub(crate) interner: DbInterner<'db>,
    pub(crate) trait_env: Arc<TraitEnvironment<'db>>,
    pub(crate) tait_coercion_table: Option<FxHashMap<OpaqueTyId, Ty>>,
    pub(crate) infer_ctxt: InferCtxt<'db>,
    pub(super) fulfillment_cx: FulfillmentCtxt<'db>,
    pub(super) diverging_type_vars: FxHashSet<crate::next_solver::Ty<'db>>,
}

pub(crate) struct InferenceTableSnapshot<'db> {
    ctxt_snapshot: CombinedSnapshot,
    obligations: FulfillmentCtxt<'db>,
}

impl<'db> InferenceTable<'db> {
    pub(crate) fn new(db: &'db dyn HirDatabase, trait_env: Arc<TraitEnvironment<'db>>) -> Self {
        let interner = DbInterner::new_with(db, Some(trait_env.krate), trait_env.block);
        let infer_ctxt = interner.infer_ctxt().build(rustc_type_ir::TypingMode::Analysis {
            defining_opaque_types_and_generators: SolverDefIds::new_from_iter(interner, []),
        });
        InferenceTable {
            db,
            interner,
            trait_env,
            tait_coercion_table: None,
            fulfillment_cx: FulfillmentCtxt::new(&infer_ctxt),
            infer_ctxt,
            diverging_type_vars: FxHashSet::default(),
        }
    }

    pub(crate) fn type_var_is_sized(&self, self_ty: TyVid) -> bool {
        let Some(sized_did) = LangItem::Sized.resolve_trait(self.db, self.trait_env.krate) else {
            return true;
        };
        self.obligations_for_self_ty(self_ty).into_iter().any(|obligation| {
            match obligation.predicate.kind().skip_binder() {
                crate::next_solver::PredicateKind::Clause(
                    crate::next_solver::ClauseKind::Trait(data),
                ) => data.def_id().0 == sized_did,
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

    fn type_matches_expected_vid(
        &self,
        expected_vid: TyVid,
        ty: crate::next_solver::Ty<'db>,
    ) -> bool {
        let ty = self.shallow_resolve(ty);

        match ty.kind() {
            crate::next_solver::TyKind::Infer(rustc_type_ir::TyVar(found_vid)) => {
                self.infer_ctxt.root_var(expected_vid) == self.infer_ctxt.root_var(found_vid)
            }
            _ => false,
        }
    }

    pub(super) fn set_diverging(&mut self, ty: crate::next_solver::Ty<'db>) {
        self.diverging_type_vars.insert(ty);
    }

    pub(crate) fn canonicalize<T>(&mut self, t: T) -> rustc_type_ir::Canonical<DbInterner<'db>, T>
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        // try to resolve obligations before canonicalizing, since this might
        // result in new knowledge about variables
        self.select_obligations_where_possible();
        self.infer_ctxt.canonicalize_response(t)
    }

    /// Recurses through the given type, normalizing associated types mentioned
    /// in it by replacing them by type variables and registering obligations to
    /// resolve later. This should be done once for every type we get from some
    /// type annotation (e.g. from a let type annotation, field type or function
    /// call). `make_ty` handles this already, but e.g. for field types we need
    /// to do it as well.
    pub(crate) fn normalize_associated_types_in<T, U>(&mut self, ty: T) -> T
    where
        T: ChalkToNextSolver<'db, U>,
        U: NextSolverToChalk<'db, T> + rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        self.normalize_associated_types_in_ns(ty.to_nextsolver(self.interner))
            .to_chalk(self.interner)
    }

    // FIXME: We should get rid of this method. We cannot deeply normalize during inference, only when finishing.
    // Inference should use shallow normalization (`try_structurally_resolve_type()`) only, when needed.
    pub(crate) fn normalize_associated_types_in_ns<T>(&mut self, ty: T) -> T
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'db>> + Clone,
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
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        fn eagerly_resolve_ty<const N: usize>(
            table: &mut InferenceTable<'_>,
            ty: Ty,
            mut tys: SmallVec<[Ty; N]>,
        ) -> Ty {
            if tys.contains(&ty) {
                return ty;
            }
            tys.push(ty.clone());

            match ty.kind(Interner) {
                TyKind::Alias(AliasTy::Projection(proj_ty)) => {
                    let ty = table.normalize_projection_ty(proj_ty.clone());
                    eagerly_resolve_ty(table, ty, tys)
                }
                TyKind::InferenceVar(..) => {
                    let ty = table.resolve_ty_shallow(&ty);
                    eagerly_resolve_ty(table, ty, tys)
                }
                _ => ty,
            }
        }

        fold_tys_and_consts(
            ty,
            |e, _| match e {
                Either::Left(ty) => {
                    Either::Left(eagerly_resolve_ty::<8>(self, ty, SmallVec::new()))
                }
                Either::Right(c) => Either::Right(match &c.data(Interner).value {
                    chalk_ir::ConstValue::Concrete(cc) => match &cc.interned {
                        crate::ConstScalar::UnevaluatedConst(c_id, subst) => {
                            // FIXME: same as `normalize_associated_types_in`
                            if subst.len(Interner) == 0 {
                                if let Ok(eval) = self.db.const_eval(*c_id, subst.clone(), None) {
                                    eval
                                } else {
                                    unknown_const(c.data(Interner).ty.clone())
                                }
                            } else {
                                unknown_const(c.data(Interner).ty.clone())
                            }
                        }
                        _ => c,
                    },
                    _ => c,
                }),
            },
            DebruijnIndex::INNERMOST,
        )
    }

    pub(crate) fn normalize_projection_ty(&mut self, proj_ty: ProjectionTy) -> Ty {
        let ty = TyKind::Alias(chalk_ir::AliasTy::Projection(proj_ty))
            .intern(Interner)
            .to_nextsolver(self.interner);
        self.normalize_alias_ty(ty).to_chalk(self.interner)
    }

    pub(crate) fn normalize_alias_ty(
        &mut self,
        alias: crate::next_solver::Ty<'db>,
    ) -> crate::next_solver::Ty<'db> {
        let infer_term = self.infer_ctxt.next_ty_var();
        let obligation = crate::next_solver::Predicate::new(
            self.interner,
            crate::next_solver::Binder::dummy(crate::next_solver::PredicateKind::AliasRelate(
                alias.into(),
                infer_term.into(),
                rustc_type_ir::AliasRelationDirection::Equate,
            )),
        );
        self.register_obligation(obligation);
        self.resolve_vars_with_obligations(infer_term)
    }

    fn new_var(&mut self, kind: TyVariableKind, diverging: bool) -> Ty {
        let var = match kind {
            TyVariableKind::General => {
                let var = self.infer_ctxt.next_ty_vid();
                InferenceVar::from(var.as_u32())
            }
            TyVariableKind::Integer => {
                let var = self.infer_ctxt.next_int_vid();
                InferenceVar::from(var.as_u32())
            }
            TyVariableKind::Float => {
                let var = self.infer_ctxt.next_float_vid();
                InferenceVar::from(var.as_u32())
            }
        };

        let ty = var.to_ty(Interner, kind);
        if diverging {
            self.diverging_type_vars.insert(ty.to_nextsolver(self.interner));
        }
        ty
    }

    pub(crate) fn new_type_var(&mut self) -> Ty {
        self.new_var(TyVariableKind::General, false)
    }

    pub(crate) fn next_ty_var(&mut self) -> crate::next_solver::Ty<'db> {
        self.infer_ctxt.next_ty_var()
    }

    pub(crate) fn new_integer_var(&mut self) -> Ty {
        self.new_var(TyVariableKind::Integer, false)
    }

    pub(crate) fn new_float_var(&mut self) -> Ty {
        self.new_var(TyVariableKind::Float, false)
    }

    pub(crate) fn new_maybe_never_var(&mut self) -> Ty {
        self.new_var(TyVariableKind::General, true)
    }

    pub(crate) fn new_const_var(&mut self, ty: Ty) -> Const {
        let var = self.infer_ctxt.next_const_vid();
        let var = InferenceVar::from(var.as_u32());
        var.to_const(Interner, ty)
    }

    pub(crate) fn new_lifetime_var(&mut self) -> Lifetime {
        let var = self.infer_ctxt.next_region_vid();
        let var = InferenceVar::from(var.as_u32());
        var.to_lifetime(Interner)
    }

    pub(crate) fn next_region_var(&mut self) -> crate::next_solver::Region<'db> {
        self.infer_ctxt.next_region_var()
    }

    pub(crate) fn resolve_with_fallback<T>(
        &mut self,
        t: T,
        fallback: &dyn Fn(InferenceVar, VariableKind, DebruijnIndex) -> GenericArg,
    ) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        self.resolve_with_fallback_inner(t, &fallback)
    }

    pub(crate) fn fresh_subst(&mut self, binders: &[CanonicalVarKind<Interner>]) -> Substitution {
        Substitution::from_iter(
            Interner,
            binders.iter().map(|kind| match &kind.kind {
                chalk_ir::VariableKind::Ty(ty_variable_kind) => {
                    self.new_var(*ty_variable_kind, false).cast(Interner)
                }
                chalk_ir::VariableKind::Lifetime => self.new_lifetime_var().cast(Interner),
                chalk_ir::VariableKind::Const(ty) => self.new_const_var(ty.clone()).cast(Interner),
            }),
        )
    }

    pub(crate) fn instantiate_canonical<T>(&mut self, canonical: Canonical<T>) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner> + std::fmt::Debug,
    {
        let subst = self.fresh_subst(canonical.binders.as_slice(Interner));
        subst.apply(canonical.value, Interner)
    }

    pub(crate) fn instantiate_canonical_ns<T>(
        &mut self,
        canonical: rustc_type_ir::Canonical<DbInterner<'db>, T>,
    ) -> T
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        self.infer_ctxt.instantiate_canonical(&canonical).0
    }

    fn resolve_with_fallback_inner<T>(
        &mut self,
        t: T,
        fallback: &dyn Fn(InferenceVar, VariableKind, DebruijnIndex) -> GenericArg,
    ) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        let var_stack = &mut vec![];
        t.fold_with(
            &mut resolve::Resolver { table: self, var_stack, fallback },
            DebruijnIndex::INNERMOST,
        )
    }

    pub(crate) fn resolve_completely<T, U>(&mut self, t: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner> + ChalkToNextSolver<'db, U>,
        U: NextSolverToChalk<'db, T> + rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        let value = t.to_nextsolver(self.interner);
        let value = self.infer_ctxt.resolve_vars_if_possible(value);

        let mut goals = vec![];
        let value = value.fold_with(&mut resolve_completely::Resolver::new(self, true, &mut goals));

        // FIXME(next-solver): Handle `goals`.

        value.to_chalk(self.interner)
    }

    /// Unify two relatable values (e.g. `Ty`) and register new trait goals that arise from that.
    pub(crate) fn unify<T: ChalkToNextSolver<'db, U>, U: Relate<DbInterner<'db>>>(
        &mut self,
        ty1: &T,
        ty2: &T,
    ) -> bool {
        let result = match self.try_unify(ty1, ty2) {
            Ok(r) => r,
            Err(_) => return false,
        };
        self.register_obligations(result.goals);
        true
    }

    pub(crate) fn unify_ns<T: Relate<DbInterner<'db>>>(&mut self, lhs: T, rhs: T) -> bool {
        let Ok(infer_ok) = self.try_unify_ns(lhs, rhs) else {
            return false;
        };
        self.register_obligations(infer_ok.goals);
        true
    }

    /// Unify two relatable values (e.g. `Ty`) and check whether trait goals which arise from that could be fulfilled
    pub(crate) fn unify_deeply<T: ChalkToNextSolver<'db, U>, U: Relate<DbInterner<'db>>>(
        &mut self,
        ty1: &T,
        ty2: &T,
    ) -> bool {
        let result = match self.try_unify(ty1, ty2) {
            Ok(r) => r,
            Err(_) => return false,
        };
        result.goals.into_iter().all(|goal| {
            matches!(next_trait_solve_in_ctxt(&self.infer_ctxt, goal), Ok((_, Certainty::Yes)))
        })
    }

    /// Unify two relatable values (e.g. `Ty`) and return new trait goals arising from it, so the
    /// caller needs to deal with them.
    pub(crate) fn try_unify<T: ChalkToNextSolver<'db, U>, U: Relate<DbInterner<'db>>>(
        &mut self,
        t1: &T,
        t2: &T,
    ) -> InferResult<'db, ()> {
        let lhs = t1.to_nextsolver(self.interner);
        let rhs = t2.to_nextsolver(self.interner);
        self.try_unify_ns(lhs, rhs)
    }

    /// Unify two relatable values (e.g. `Ty`) and return new trait goals arising from it, so the
    /// caller needs to deal with them.
    pub(crate) fn try_unify_ns<T: Relate<DbInterner<'db>>>(
        &mut self,
        lhs: T,
        rhs: T,
    ) -> InferResult<'db, ()> {
        let variance = rustc_type_ir::Variance::Invariant;
        let span = crate::next_solver::Span::dummy();
        match self.infer_ctxt.relate(self.trait_env.env, lhs, variance, rhs, span) {
            Ok(goals) => Ok(crate::infer::InferOk { goals, value: () }),
            Err(_) => Err(TypeError),
        }
    }

    /// If `ty` is a type variable with known type, returns that type;
    /// otherwise, return ty.
    #[tracing::instrument(skip(self))]
    pub(crate) fn resolve_ty_shallow(&mut self, ty: &Ty) -> Ty {
        self.shallow_resolve(ty.to_nextsolver(self.interner)).to_chalk(self.interner)
    }

    pub(crate) fn shallow_resolve(
        &self,
        ty: crate::next_solver::Ty<'db>,
    ) -> crate::next_solver::Ty<'db> {
        self.infer_ctxt.shallow_resolve(ty)
    }

    pub(crate) fn resolve_vars_with_obligations<T>(&mut self, t: T) -> T
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        use rustc_type_ir::TypeVisitableExt;

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

    pub(crate) fn structurally_resolve_type(&mut self, ty: &Ty) -> Ty {
        if let TyKind::Alias(chalk_ir::AliasTy::Projection(..)) = ty.kind(Interner) {
            self.structurally_normalize_ty(ty)
        } else {
            self.resolve_vars_with_obligations(ty.to_nextsolver(self.interner))
                .to_chalk(self.interner)
        }
    }

    fn structurally_normalize_ty(&mut self, ty: &Ty) -> Ty {
        self.structurally_normalize_term(ty.to_nextsolver(self.interner).into())
            .expect_ty()
            .to_chalk(self.interner)
    }

    fn structurally_normalize_term(&mut self, term: Term<'db>) -> Term<'db> {
        self.infer_ctxt
            .at(&ObligationCause::new(), self.trait_env.env)
            .structurally_normalize_term(term, &mut self.fulfillment_cx)
            .unwrap_or(term)
    }

    /// Try to resolve `ty` to a structural type, normalizing aliases.
    ///
    /// In case there is still ambiguity, the returned type may be an inference
    /// variable. This is different from `structurally_resolve_type` which errors
    /// in this case.
    pub(crate) fn try_structurally_resolve_type(
        &mut self,
        ty: crate::next_solver::Ty<'db>,
    ) -> crate::next_solver::Ty<'db> {
        if let crate::next_solver::TyKind::Alias(..) = ty.kind() {
            // We need to use a separate variable here as otherwise the temporary for
            // `self.fulfillment_cx.borrow_mut()` is alive in the `Err` branch, resulting
            // in a reentrant borrow, causing an ICE.
            let result = self
                .infer_ctxt
                .at(&ObligationCause::misc(), self.trait_env.env)
                .structurally_normalize_ty(ty, &mut self.fulfillment_cx);
            match result {
                Ok(normalized_ty) => normalized_ty,
                Err(_errors) => crate::next_solver::Ty::new_error(self.interner, ErrorGuaranteed),
            }
        } else {
            self.resolve_vars_with_obligations(ty)
        }
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
                        self.interner,
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

    pub(crate) fn register_obligations(
        &mut self,
        obligations: Vec<crate::next_solver::Goal<'db, crate::next_solver::Predicate<'db>>>,
    ) {
        obligations.into_iter().for_each(|goal| self.register_obligation_in_env(goal));
    }

    pub(crate) fn select_obligations_where_possible(&mut self) {
        self.fulfillment_cx.select_where_possible(&self.infer_ctxt);
    }

    pub(super) fn register_predicate(
        &mut self,
        obligation: crate::next_solver::infer::traits::PredicateObligation<'db>,
    ) {
        if obligation.has_escaping_bound_vars() {
            panic!("escaping bound vars in predicate {:?}", obligation);
        }

        self.fulfillment_cx.register_predicate_obligation(&self.infer_ctxt, obligation);
    }

    pub(super) fn register_predicates<I>(&mut self, obligations: I)
    where
        I: IntoIterator<Item = crate::next_solver::infer::traits::PredicateObligation<'db>>,
    {
        obligations.into_iter().for_each(|obligation| {
            self.register_predicate(obligation);
        });
    }

    pub(crate) fn callable_sig(
        &mut self,
        ty: &Ty,
        num_args: usize,
    ) -> Option<(Option<FnTrait>, Vec<crate::next_solver::Ty<'db>>, crate::next_solver::Ty<'db>)>
    {
        match ty.callable_sig(self.db) {
            Some(sig) => Some((
                None,
                sig.params().iter().map(|param| param.to_nextsolver(self.interner)).collect(),
                sig.ret().to_nextsolver(self.interner),
            )),
            None => {
                let (f, args_ty, return_ty) = self.callable_sig_from_fn_trait(ty, num_args)?;
                Some((Some(f), args_ty, return_ty))
            }
        }
    }

    fn callable_sig_from_fn_trait(
        &mut self,
        ty: &Ty,
        num_args: usize,
    ) -> Option<(FnTrait, Vec<next_solver::Ty<'db>>, next_solver::Ty<'db>)> {
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
            let arg_ty = next_solver::Ty::new_tup_from_iter(
                self.interner,
                std::iter::repeat_with(|| {
                    let ty = self.next_ty_var();
                    arg_tys.push(ty);
                    ty
                })
                .take(num_args),
            );
            let args = [ty.to_nextsolver(self.interner), arg_ty];
            let trait_ref = crate::next_solver::TraitRef::new(self.interner, fn_trait.into(), args);

            let projection = crate::next_solver::Ty::new_alias(
                self.interner,
                rustc_type_ir::AliasTyKind::Projection,
                crate::next_solver::AliasTy::new(self.interner, output_assoc_type.into(), args),
            );

            let pred = crate::next_solver::Predicate::upcast_from(trait_ref, self.interner);
            if !self.try_obligation(pred).no_solution() {
                self.register_obligation(pred);
                let return_ty = self.normalize_alias_ty(projection);
                for &fn_x in subtraits {
                    let fn_x_trait = fn_x.get_id(self.db, krate)?;
                    let trait_ref =
                        crate::next_solver::TraitRef::new(self.interner, fn_x_trait.into(), args);
                    let pred = crate::next_solver::Predicate::upcast_from(trait_ref, self.interner);
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
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        fold_generic_args(
            ty,
            |arg, _| match arg {
                GenericArgData::Ty(ty) => GenericArgData::Ty(self.insert_type_vars_shallow(ty)),
                // FIXME: insert lifetime vars once LifetimeData::InferenceVar
                // and specific error variant for lifetimes start being constructed
                GenericArgData::Lifetime(lt) => GenericArgData::Lifetime(lt),
                GenericArgData::Const(c) => {
                    GenericArgData::Const(self.insert_const_vars_shallow(c))
                }
            },
            DebruijnIndex::INNERMOST,
        )
    }

    /// Replaces `Ty::Error` by a new type var, so we can maybe still infer it.
    pub(super) fn insert_type_vars_shallow(&mut self, ty: Ty) -> Ty {
        match ty.kind(Interner) {
            TyKind::Error => self.new_type_var(),
            TyKind::InferenceVar(..) => {
                let ty_resolved = self.structurally_resolve_type(&ty);
                if ty_resolved.is_unknown() { self.new_type_var() } else { ty }
            }
            _ => ty,
        }
    }

    /// Whenever you lower a user-written type, you should call this.
    pub(crate) fn process_user_written_ty<T, U>(&mut self, ty: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner> + ChalkToNextSolver<'db, U>,
        U: NextSolverToChalk<'db, T> + rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        self.process_remote_user_written_ty(ty)
        // FIXME: Register a well-formed obligation.
    }

    /// The difference of this method from `process_user_written_ty()` is that this method doesn't register a well-formed obligation,
    /// while `process_user_written_ty()` should (but doesn't currently).
    pub(crate) fn process_remote_user_written_ty<T, U>(&mut self, ty: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner> + ChalkToNextSolver<'db, U>,
        U: NextSolverToChalk<'db, T> + rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        let ty = self.insert_type_vars(ty);
        // See https://github.com/rust-lang/rust/blob/cdb45c87e2cd43495379f7e867e3cc15dcee9f93/compiler/rustc_hir_typeck/src/fn_ctxt/mod.rs#L487-L495:
        // Even though the new solver only lazily normalizes usually, here we eagerly normalize so that not everything needs
        // to normalize before inspecting the `TyKind`.
        // FIXME(next-solver): We should not deeply normalize here, only shallowly.
        self.normalize_associated_types_in(ty)
    }

    /// Replaces ConstScalar::Unknown by a new type var, so we can maybe still infer it.
    pub(super) fn insert_const_vars_shallow(&mut self, c: Const) -> Const {
        let data = c.data(Interner);
        match &data.value {
            ConstValue::Concrete(cc) => match &cc.interned {
                crate::ConstScalar::Unknown => self.new_const_var(data.ty.clone()),
                // try to evaluate unevaluated const. Replace with new var if const eval failed.
                crate::ConstScalar::UnevaluatedConst(id, subst) => {
                    if let Ok(eval) = self.db.const_eval(*id, subst.clone(), None) {
                        eval
                    } else {
                        self.new_const_var(data.ty.clone())
                    }
                }
                _ => c,
            },
            _ => c,
        }
    }

    /// Check if given type is `Sized` or not
    pub(crate) fn is_sized(&mut self, ty: &Ty) -> bool {
        fn short_circuit_trivial_tys(ty: &Ty) -> Option<bool> {
            match ty.kind(Interner) {
                TyKind::Scalar(..)
                | TyKind::Ref(..)
                | TyKind::Raw(..)
                | TyKind::Never
                | TyKind::FnDef(..)
                | TyKind::Array(..)
                | TyKind::Function(..) => Some(true),
                TyKind::Slice(..) | TyKind::Str | TyKind::Dyn(..) => Some(false),
                _ => None,
            }
        }

        let mut ty = ty.clone();
        ty = self.eagerly_normalize_and_resolve_shallow_in(ty);
        if let Some(sized) = short_circuit_trivial_tys(&ty) {
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
                        .clone()
                        .substitute(Interner, subst);
                    if structs.contains(&ty) {
                        // A struct recursively contains itself as a tail field somewhere.
                        return true; // Don't overload the users with too many errors.
                    }
                    structs.push(ty);
                    // Structs can have DST as its last field and such cases are not handled
                    // as unsized by the chalk, so we do this manually.
                    ty = last_field_ty;
                    ty = self.eagerly_normalize_and_resolve_shallow_in(ty);
                    if let Some(sized) = short_circuit_trivial_tys(&ty) {
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
            TraitRef::new(self.interner, sized.into(), [ty.to_nextsolver(self.interner)]),
            self.interner,
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

mod resolve {
    use super::InferenceTable;
    use crate::{
        Const, DebruijnIndex, GenericArg, InferenceVar, Interner, Lifetime, Ty, TyVariableKind,
        VariableKind, next_solver::mapping::NextSolverToChalk,
    };
    use chalk_ir::fold::{TypeFoldable, TypeFolder};
    use rustc_type_ir::{FloatVid, IntVid, TyVid};

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub(super) enum VarKind {
        Ty(TyVariableKind),
        Const,
    }

    #[derive(chalk_derive::FallibleTypeFolder)]
    #[has_interner(Interner)]
    pub(super) struct Resolver<
        'a,
        'b,
        F: Fn(InferenceVar, VariableKind, DebruijnIndex) -> GenericArg,
    > {
        pub(super) table: &'a mut InferenceTable<'b>,
        pub(super) var_stack: &'a mut Vec<(InferenceVar, VarKind)>,
        pub(super) fallback: F,
    }
    impl<F> TypeFolder<Interner> for Resolver<'_, '_, F>
    where
        F: Fn(InferenceVar, VariableKind, DebruijnIndex) -> GenericArg,
    {
        fn as_dyn(&mut self) -> &mut dyn TypeFolder<Interner> {
            self
        }

        fn interner(&self) -> Interner {
            Interner
        }

        fn fold_inference_ty(
            &mut self,
            var: InferenceVar,
            kind: TyVariableKind,
            outer_binder: DebruijnIndex,
        ) -> Ty {
            match kind {
                TyVariableKind::General => {
                    let vid = self.table.infer_ctxt.root_var(TyVid::from(var.index()));
                    let var = InferenceVar::from(vid.as_u32());
                    if self.var_stack.contains(&(var, VarKind::Ty(kind))) {
                        // recursive type
                        return (self.fallback)(var, VariableKind::Ty(kind), outer_binder)
                            .assert_ty_ref(Interner)
                            .clone();
                    }
                    if let Ok(known_ty) = self.table.infer_ctxt.probe_ty_var(vid) {
                        let known_ty: Ty = known_ty.to_chalk(self.table.interner);
                        // known_ty may contain other variables that are known by now
                        self.var_stack.push((var, VarKind::Ty(kind)));
                        let result = known_ty.fold_with(self, outer_binder);
                        self.var_stack.pop();
                        result
                    } else {
                        (self.fallback)(var, VariableKind::Ty(kind), outer_binder)
                            .assert_ty_ref(Interner)
                            .clone()
                    }
                }
                TyVariableKind::Integer => {
                    let vid = self
                        .table
                        .infer_ctxt
                        .inner
                        .borrow_mut()
                        .int_unification_table()
                        .find(IntVid::from(var.index()));
                    let var = InferenceVar::from(vid.as_u32());
                    if self.var_stack.contains(&(var, VarKind::Ty(kind))) {
                        // recursive type
                        return (self.fallback)(var, VariableKind::Ty(kind), outer_binder)
                            .assert_ty_ref(Interner)
                            .clone();
                    }
                    if let Some(known_ty) = self.table.infer_ctxt.resolve_int_var(vid) {
                        let known_ty: Ty = known_ty.to_chalk(self.table.interner);
                        // known_ty may contain other variables that are known by now
                        self.var_stack.push((var, VarKind::Ty(kind)));
                        let result = known_ty.fold_with(self, outer_binder);
                        self.var_stack.pop();
                        result
                    } else {
                        (self.fallback)(var, VariableKind::Ty(kind), outer_binder)
                            .assert_ty_ref(Interner)
                            .clone()
                    }
                }
                TyVariableKind::Float => {
                    let vid = self
                        .table
                        .infer_ctxt
                        .inner
                        .borrow_mut()
                        .float_unification_table()
                        .find(FloatVid::from(var.index()));
                    let var = InferenceVar::from(vid.as_u32());
                    if self.var_stack.contains(&(var, VarKind::Ty(kind))) {
                        // recursive type
                        return (self.fallback)(var, VariableKind::Ty(kind), outer_binder)
                            .assert_ty_ref(Interner)
                            .clone();
                    }
                    if let Some(known_ty) = self.table.infer_ctxt.resolve_float_var(vid) {
                        let known_ty: Ty = known_ty.to_chalk(self.table.interner);
                        // known_ty may contain other variables that are known by now
                        self.var_stack.push((var, VarKind::Ty(kind)));
                        let result = known_ty.fold_with(self, outer_binder);
                        self.var_stack.pop();
                        result
                    } else {
                        (self.fallback)(var, VariableKind::Ty(kind), outer_binder)
                            .assert_ty_ref(Interner)
                            .clone()
                    }
                }
            }
        }

        fn fold_inference_const(
            &mut self,
            ty: Ty,
            var: InferenceVar,
            outer_binder: DebruijnIndex,
        ) -> Const {
            let vid = self
                .table
                .infer_ctxt
                .root_const_var(rustc_type_ir::ConstVid::from_u32(var.index()));
            let var = InferenceVar::from(vid.as_u32());
            if self.var_stack.contains(&(var, VarKind::Const)) {
                // recursive
                return (self.fallback)(var, VariableKind::Const(ty), outer_binder)
                    .assert_const_ref(Interner)
                    .clone();
            }
            if let Ok(known_const) = self.table.infer_ctxt.probe_const_var(vid) {
                let known_const: Const = known_const.to_chalk(self.table.interner);
                // known_ty may contain other variables that are known by now
                self.var_stack.push((var, VarKind::Const));
                let result = known_const.fold_with(self, outer_binder);
                self.var_stack.pop();
                result
            } else {
                (self.fallback)(var, VariableKind::Const(ty), outer_binder)
                    .assert_const_ref(Interner)
                    .clone()
            }
        }

        fn fold_inference_lifetime(
            &mut self,
            _var: InferenceVar,
            _outer_binder: DebruijnIndex,
        ) -> Lifetime {
            // fall back all lifetimes to 'error -- currently we don't deal
            // with any lifetimes, but we can sometimes get some lifetime
            // variables through Chalk's unification, and this at least makes
            // sure we don't leak them outside of inference
            crate::error_lifetime()
        }
    }
}

mod resolve_completely {
    use rustc_type_ir::{
        DebruijnIndex, Flags, TypeFolder, TypeSuperFoldable,
        inherent::{Const as _, Ty as _},
    };

    use crate::next_solver::Region;
    use crate::{
        infer::unify::InferenceTable,
        next_solver::{
            Const, DbInterner, ErrorGuaranteed, Goal, Predicate, Term, Ty,
            infer::traits::ObligationCause,
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

            value.fold_with(&mut ReplaceInferWithError { interner: self.ctx.interner })
        }
    }

    impl<'cx, 'db> TypeFolder<DbInterner<'db>> for Resolver<'cx, 'db> {
        fn cx(&self) -> DbInterner<'db> {
            self.ctx.interner
        }

        fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
            if r.is_var() { Region::error(self.ctx.interner) } else { r }
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

    struct ReplaceInferWithError<'db> {
        interner: DbInterner<'db>,
    }

    impl<'db> TypeFolder<DbInterner<'db>> for ReplaceInferWithError<'db> {
        fn cx(&self) -> DbInterner<'db> {
            self.interner
        }

        fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
            if t.is_infer() {
                Ty::new_error(self.interner, ErrorGuaranteed)
            } else {
                t.super_fold_with(self)
            }
        }

        fn fold_const(&mut self, c: Const<'db>) -> Const<'db> {
            if c.is_ct_infer() {
                Const::new_error(self.interner, ErrorGuaranteed)
            } else {
                c.super_fold_with(self)
            }
        }

        fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
            if r.is_var() { Region::error(self.interner) } else { r }
        }
    }
}

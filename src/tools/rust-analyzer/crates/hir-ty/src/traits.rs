//! Trait solving using next trait solver.

use core::fmt;
use std::hash::Hash;

use chalk_ir::{DebruijnIndex, GoalData, fold::TypeFoldable};

use base_db::Crate;
use hir_def::{BlockId, TraitId, lang_item::LangItem};
use hir_expand::name::Name;
use intern::sym;
use rustc_next_trait_solver::solve::{HasChanged, SolverDelegateEvalExt};
use rustc_type_ir::{
    InferCtxtLike, TypingMode,
    inherent::{IntoKind, SliceLike, Span as _, Ty as _},
    solve::Certainty,
};
use span::Edition;
use stdx::never;
use triomphe::Arc;

use crate::{
    AliasEq, AliasTy, Canonical, DomainGoal, Goal, InEnvironment, Interner, ProjectionTy,
    ProjectionTyExt, TraitRefExt, Ty, TyKind, TypeFlags, WhereClause,
    db::HirDatabase,
    from_assoc_type_id,
    next_solver::{
        DbInterner, GenericArg, ParamEnv, Predicate, SolverContext, Span,
        infer::{DbInternerInferExt, InferCtxt, traits::ObligationCause},
        mapping::{ChalkToNextSolver, NextSolverToChalk, convert_canonical_args_for_result},
        obligation_ctxt::ObligationCtxt,
        util::mini_canonicalize,
    },
    utils::UnevaluatedConstEvaluatorFolder,
};

/// A set of clauses that we assume to be true. E.g. if we are inside this function:
/// ```rust
/// fn foo<T: Default>(t: T) {}
/// ```
/// we assume that `T: Default`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraitEnvironment<'db> {
    pub krate: Crate,
    pub block: Option<BlockId>,
    // FIXME make this a BTreeMap
    traits_from_clauses: Box<[(crate::next_solver::Ty<'db>, TraitId)]>,
    pub env: ParamEnv<'db>,
}

impl<'db> TraitEnvironment<'db> {
    pub fn empty(krate: Crate) -> Arc<Self> {
        Arc::new(TraitEnvironment {
            krate,
            block: None,
            traits_from_clauses: Box::default(),
            env: ParamEnv::empty(),
        })
    }

    pub fn new(
        krate: Crate,
        block: Option<BlockId>,
        traits_from_clauses: Box<[(crate::next_solver::Ty<'db>, TraitId)]>,
        env: ParamEnv<'db>,
    ) -> Arc<Self> {
        Arc::new(TraitEnvironment { krate, block, traits_from_clauses, env })
    }

    // pub fn with_block(self: &mut Arc<Self>, block: BlockId) {
    pub fn with_block(this: &mut Arc<Self>, block: BlockId) {
        Arc::make_mut(this).block = Some(block);
    }

    pub fn traits_in_scope_from_clauses(
        &self,
        ty: crate::next_solver::Ty<'db>,
    ) -> impl Iterator<Item = TraitId> + '_ {
        self.traits_from_clauses
            .iter()
            .filter_map(move |(self_ty, trait_id)| (*self_ty == ty).then_some(*trait_id))
    }
}

/// This should be used in `hir` only.
pub fn structurally_normalize_ty<'db>(
    infcx: &InferCtxt<'db>,
    ty: crate::next_solver::Ty<'db>,
    env: Arc<TraitEnvironment<'db>>,
) -> crate::next_solver::Ty<'db> {
    let crate::next_solver::TyKind::Alias(..) = ty.kind() else { return ty };
    let mut ocx = ObligationCtxt::new(infcx);
    let ty = ocx.structurally_normalize_ty(&ObligationCause::dummy(), env.env, ty).unwrap_or(ty);
    ty.replace_infer_with_error(infcx.interner)
}

pub(crate) fn normalize_projection_query<'db>(
    db: &'db dyn HirDatabase,
    projection: ProjectionTy,
    env: Arc<TraitEnvironment<'db>>,
) -> Ty {
    if projection.substitution.iter(Interner).any(|arg| {
        arg.ty(Interner)
            .is_some_and(|ty| ty.data(Interner).flags.intersects(TypeFlags::HAS_TY_INFER))
    }) {
        never!(
            "Invoking `normalize_projection_query` with a projection type containing inference var"
        );
        return TyKind::Error.intern(Interner);
    }

    let interner = DbInterner::new_with(db, Some(env.krate), env.block);
    // FIXME(next-solver): I believe this should use `PostAnalysis` (this is only used for IDE things),
    // but this causes some bug because of our incorrect impl of `type_of_opaque_hir_typeck()` for TAIT
    // and async blocks.
    let infcx = interner.infer_ctxt().build(TypingMode::Analysis {
        defining_opaque_types_and_generators: crate::next_solver::SolverDefIds::new_from_iter(
            interner,
            [],
        ),
    });
    let alias_ty = crate::next_solver::Ty::new_alias(
        interner,
        rustc_type_ir::AliasTyKind::Projection,
        crate::next_solver::AliasTy::new(
            interner,
            from_assoc_type_id(projection.associated_ty_id).into(),
            <crate::Substitution as ChalkToNextSolver<crate::next_solver::GenericArgs<'_>>>::to_nextsolver(&projection.substitution, interner),
        ),
    );
    let mut ctxt = crate::next_solver::obligation_ctxt::ObligationCtxt::new(&infcx);
    let normalized = ctxt
        .structurally_normalize_ty(&ObligationCause::dummy(), env.env, alias_ty)
        .unwrap_or(alias_ty);
    normalized.replace_infer_with_error(interner).to_chalk(interner)
}

fn identity_subst(
    binders: chalk_ir::CanonicalVarKinds<Interner>,
) -> chalk_ir::Canonical<chalk_ir::Substitution<Interner>> {
    let identity_subst = chalk_ir::Substitution::from_iter(
        Interner,
        binders.iter(Interner).enumerate().map(|(index, c)| {
            let index_db = chalk_ir::BoundVar::new(DebruijnIndex::INNERMOST, index);
            match &c.kind {
                chalk_ir::VariableKind::Ty(_) => {
                    chalk_ir::GenericArgData::Ty(TyKind::BoundVar(index_db).intern(Interner))
                        .intern(Interner)
                }
                chalk_ir::VariableKind::Lifetime => chalk_ir::GenericArgData::Lifetime(
                    chalk_ir::LifetimeData::BoundVar(index_db).intern(Interner),
                )
                .intern(Interner),
                chalk_ir::VariableKind::Const(ty) => chalk_ir::GenericArgData::Const(
                    chalk_ir::ConstData {
                        ty: ty.clone(),
                        value: chalk_ir::ConstValue::BoundVar(index_db),
                    }
                    .intern(Interner),
                )
                .intern(Interner),
            }
        }),
    );
    chalk_ir::Canonical { binders, value: identity_subst }
}

/// Solve a trait goal using next trait solver.
pub(crate) fn trait_solve_query(
    db: &dyn HirDatabase,
    krate: Crate,
    block: Option<BlockId>,
    goal: Canonical<InEnvironment<Goal>>,
) -> NextTraitSolveResult {
    let _p = tracing::info_span!("trait_solve_query", detail = ?match &goal.value.goal.data(Interner) {
        GoalData::DomainGoal(DomainGoal::Holds(WhereClause::Implemented(it))) => db
            .trait_signature(it.hir_trait_id())
            .name
            .display(db, Edition::LATEST)
            .to_string(),
        GoalData::DomainGoal(DomainGoal::Holds(WhereClause::AliasEq(_))) => "alias_eq".to_owned(),
        _ => "??".to_owned(),
    })
    .entered();

    if let GoalData::DomainGoal(DomainGoal::Holds(WhereClause::AliasEq(AliasEq {
        alias: AliasTy::Projection(projection_ty),
        ..
    }))) = &goal.value.goal.data(Interner)
        && let TyKind::BoundVar(_) = projection_ty.self_type_parameter(db).kind(Interner)
    {
        // Hack: don't ask Chalk to normalize with an unknown self type, it'll say that's impossible
        return NextTraitSolveResult::Uncertain(identity_subst(goal.binders.clone()));
    }

    // Chalk see `UnevaluatedConst` as a unique concrete value, but we see it as an alias for another const. So
    // we should get rid of it when talking to chalk.
    let goal = goal
        .try_fold_with(&mut UnevaluatedConstEvaluatorFolder { db }, DebruijnIndex::INNERMOST)
        .unwrap();

    // We currently don't deal with universes (I think / hope they're not yet
    // relevant for our use cases?)
    next_trait_solve(db, krate, block, goal)
}

fn solve_nextsolver<'db>(
    db: &'db dyn HirDatabase,
    krate: Crate,
    block: Option<BlockId>,
    goal: &chalk_ir::UCanonical<chalk_ir::InEnvironment<chalk_ir::Goal<Interner>>>,
) -> Result<
    (HasChanged, Certainty, rustc_type_ir::Canonical<DbInterner<'db>, Vec<GenericArg<'db>>>),
    rustc_type_ir::solve::NoSolution,
> {
    // FIXME: should use analysis_in_body, but that needs GenericDefId::Block
    let context = SolverContext(
        DbInterner::new_with(db, Some(krate), block)
            .infer_ctxt()
            .build(TypingMode::non_body_analysis()),
    );

    match goal.canonical.value.goal.data(Interner) {
        // FIXME: args here should be...what? not empty
        GoalData::All(goals) if goals.is_empty(Interner) => {
            return Ok((HasChanged::No, Certainty::Yes, mini_canonicalize(context, vec![])));
        }
        _ => {}
    }

    let goal = goal.canonical.to_nextsolver(context.cx());
    tracing::info!(?goal);

    let (goal, var_values) = context.instantiate_canonical(&goal);
    tracing::info!(?var_values);

    let res = context.evaluate_root_goal(goal, Span::dummy(), None);

    let vars =
        var_values.var_values.iter().map(|g| context.0.resolve_vars_if_possible(g)).collect();
    let canonical_var_values = mini_canonicalize(context, vars);

    let res = res.map(|r| (r.has_changed, r.certainty, canonical_var_values));

    tracing::debug!("solve_nextsolver({:?}) => {:?}", goal, res);

    res
}

#[derive(Clone, Debug, PartialEq)]
pub enum NextTraitSolveResult {
    Certain(chalk_ir::Canonical<chalk_ir::ConstrainedSubst<Interner>>),
    Uncertain(chalk_ir::Canonical<chalk_ir::Substitution<Interner>>),
    NoSolution,
}

impl NextTraitSolveResult {
    pub fn no_solution(&self) -> bool {
        matches!(self, NextTraitSolveResult::NoSolution)
    }

    pub fn certain(&self) -> bool {
        matches!(self, NextTraitSolveResult::Certain(..))
    }

    pub fn uncertain(&self) -> bool {
        matches!(self, NextTraitSolveResult::Uncertain(..))
    }
}

pub fn next_trait_solve(
    db: &dyn HirDatabase,
    krate: Crate,
    block: Option<BlockId>,
    goal: Canonical<InEnvironment<Goal>>,
) -> NextTraitSolveResult {
    let detail = match &goal.value.goal.data(Interner) {
        GoalData::DomainGoal(DomainGoal::Holds(WhereClause::Implemented(it))) => {
            db.trait_signature(it.hir_trait_id()).name.display(db, Edition::LATEST).to_string()
        }
        GoalData::DomainGoal(DomainGoal::Holds(WhereClause::AliasEq(_))) => "alias_eq".to_owned(),
        _ => "??".to_owned(),
    };
    let _p = tracing::info_span!("next_trait_solve", ?detail).entered();
    tracing::info!("next_trait_solve({:?})", goal.value.goal);

    if let GoalData::DomainGoal(DomainGoal::Holds(WhereClause::AliasEq(AliasEq {
        alias: AliasTy::Projection(projection_ty),
        ..
    }))) = &goal.value.goal.data(Interner)
        && let TyKind::BoundVar(_) = projection_ty.self_type_parameter(db).kind(Interner)
    {
        // Hack: don't ask Chalk to normalize with an unknown self type, it'll say that's impossible
        // FIXME
        return NextTraitSolveResult::Uncertain(identity_subst(goal.binders.clone()));
    }

    // Chalk see `UnevaluatedConst` as a unique concrete value, but we see it as an alias for another const. So
    // we should get rid of it when talking to chalk.
    let goal = goal
        .try_fold_with(&mut UnevaluatedConstEvaluatorFolder { db }, DebruijnIndex::INNERMOST)
        .unwrap();

    // We currently don't deal with universes (I think / hope they're not yet
    // relevant for our use cases?)
    let u_canonical = chalk_ir::UCanonical { canonical: goal, universes: 1 };
    tracing::info!(?u_canonical);

    let next_solver_res = solve_nextsolver(db, krate, block, &u_canonical);

    match next_solver_res {
        Err(_) => NextTraitSolveResult::NoSolution,
        Ok((_, Certainty::Yes, args)) => NextTraitSolveResult::Certain(
            convert_canonical_args_for_result(DbInterner::new_with(db, Some(krate), block), args),
        ),
        Ok((_, Certainty::Maybe { .. }, args)) => {
            let subst = convert_canonical_args_for_result(
                DbInterner::new_with(db, Some(krate), block),
                args,
            );
            NextTraitSolveResult::Uncertain(chalk_ir::Canonical {
                binders: subst.binders,
                value: subst.value.subst,
            })
        }
    }
}

pub fn next_trait_solve_canonical_in_ctxt<'db>(
    infer_ctxt: &InferCtxt<'db>,
    goal: crate::next_solver::Canonical<'db, crate::next_solver::Goal<'db, Predicate<'db>>>,
) -> NextTraitSolveResult {
    let context = SolverContext(infer_ctxt.clone());

    tracing::info!(?goal);

    let (goal, var_values) = context.instantiate_canonical(&goal);
    tracing::info!(?var_values);

    let res = context.evaluate_root_goal(goal, Span::dummy(), None);

    let vars =
        var_values.var_values.iter().map(|g| context.0.resolve_vars_if_possible(g)).collect();
    let canonical_var_values = mini_canonicalize(context, vars);

    let res = res.map(|r| (r.has_changed, r.certainty, canonical_var_values));

    tracing::debug!("solve_nextsolver({:?}) => {:?}", goal, res);

    match res {
        Err(_) => NextTraitSolveResult::NoSolution,
        Ok((_, Certainty::Yes, args)) => NextTraitSolveResult::Certain(
            convert_canonical_args_for_result(infer_ctxt.interner, args),
        ),
        Ok((_, Certainty::Maybe { .. }, args)) => {
            let subst = convert_canonical_args_for_result(infer_ctxt.interner, args);
            NextTraitSolveResult::Uncertain(chalk_ir::Canonical {
                binders: subst.binders,
                value: subst.value.subst,
            })
        }
    }
}

/// Solve a trait goal using next trait solver.
pub fn next_trait_solve_in_ctxt<'db, 'a>(
    infer_ctxt: &'a InferCtxt<'db>,
    goal: crate::next_solver::Goal<'db, crate::next_solver::Predicate<'db>>,
) -> Result<(HasChanged, Certainty), rustc_type_ir::solve::NoSolution> {
    tracing::info!(?goal);

    let context = <&SolverContext<'db>>::from(infer_ctxt);

    let res = context.evaluate_root_goal(goal, Span::dummy(), None);

    let res = res.map(|r| (r.has_changed, r.certainty));

    tracing::debug!("solve_nextsolver({:?}) => {:?}", goal, res);

    res
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FnTrait {
    // Warning: Order is important. If something implements `x` it should also implement
    // `y` if `y <= x`.
    FnOnce,
    FnMut,
    Fn,

    AsyncFnOnce,
    AsyncFnMut,
    AsyncFn,
}

impl fmt::Display for FnTrait {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FnTrait::FnOnce => write!(f, "FnOnce"),
            FnTrait::FnMut => write!(f, "FnMut"),
            FnTrait::Fn => write!(f, "Fn"),
            FnTrait::AsyncFnOnce => write!(f, "AsyncFnOnce"),
            FnTrait::AsyncFnMut => write!(f, "AsyncFnMut"),
            FnTrait::AsyncFn => write!(f, "AsyncFn"),
        }
    }
}

impl FnTrait {
    pub const fn function_name(&self) -> &'static str {
        match self {
            FnTrait::FnOnce => "call_once",
            FnTrait::FnMut => "call_mut",
            FnTrait::Fn => "call",
            FnTrait::AsyncFnOnce => "async_call_once",
            FnTrait::AsyncFnMut => "async_call_mut",
            FnTrait::AsyncFn => "async_call",
        }
    }

    const fn lang_item(self) -> LangItem {
        match self {
            FnTrait::FnOnce => LangItem::FnOnce,
            FnTrait::FnMut => LangItem::FnMut,
            FnTrait::Fn => LangItem::Fn,
            FnTrait::AsyncFnOnce => LangItem::AsyncFnOnce,
            FnTrait::AsyncFnMut => LangItem::AsyncFnMut,
            FnTrait::AsyncFn => LangItem::AsyncFn,
        }
    }

    pub const fn from_lang_item(lang_item: LangItem) -> Option<Self> {
        match lang_item {
            LangItem::FnOnce => Some(FnTrait::FnOnce),
            LangItem::FnMut => Some(FnTrait::FnMut),
            LangItem::Fn => Some(FnTrait::Fn),
            LangItem::AsyncFnOnce => Some(FnTrait::AsyncFnOnce),
            LangItem::AsyncFnMut => Some(FnTrait::AsyncFnMut),
            LangItem::AsyncFn => Some(FnTrait::AsyncFn),
            _ => None,
        }
    }

    pub fn method_name(self) -> Name {
        match self {
            FnTrait::FnOnce => Name::new_symbol_root(sym::call_once),
            FnTrait::FnMut => Name::new_symbol_root(sym::call_mut),
            FnTrait::Fn => Name::new_symbol_root(sym::call),
            FnTrait::AsyncFnOnce => Name::new_symbol_root(sym::async_call_once),
            FnTrait::AsyncFnMut => Name::new_symbol_root(sym::async_call_mut),
            FnTrait::AsyncFn => Name::new_symbol_root(sym::async_call),
        }
    }

    pub fn get_id(self, db: &dyn HirDatabase, krate: Crate) -> Option<TraitId> {
        self.lang_item().resolve_trait(db, krate)
    }
}

/// This should not be used in `hir-ty`, only in `hir`.
pub fn implements_trait_unique<'db>(
    ty: crate::next_solver::Ty<'db>,
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    trait_: TraitId,
) -> bool {
    implements_trait_unique_impl(db, env, trait_, &mut |infcx| {
        infcx.fill_rest_fresh_args(trait_.into(), [ty.into()])
    })
}

/// This should not be used in `hir-ty`, only in `hir`.
pub fn implements_trait_unique_with_args<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    trait_: TraitId,
    args: crate::next_solver::GenericArgs<'db>,
) -> bool {
    implements_trait_unique_impl(db, env, trait_, &mut |_| args)
}

fn implements_trait_unique_impl<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    trait_: TraitId,
    create_args: &mut dyn FnMut(&InferCtxt<'db>) -> crate::next_solver::GenericArgs<'db>,
) -> bool {
    let interner = DbInterner::new_with(db, Some(env.krate), env.block);
    // FIXME(next-solver): I believe this should be `PostAnalysis`.
    let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());

    let args = create_args(&infcx);
    let trait_ref = rustc_type_ir::TraitRef::new_from_args(interner, trait_.into(), args);
    let goal = crate::next_solver::Goal::new(interner, env.env, trait_ref);

    let result = crate::traits::next_trait_solve_in_ctxt(&infcx, goal);
    matches!(result, Ok((_, Certainty::Yes)))
}

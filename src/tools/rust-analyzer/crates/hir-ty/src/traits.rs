//! Trait solving using Chalk.

use core::fmt;

use chalk_ir::{DebruijnIndex, GoalData, fold::TypeFoldable};
use chalk_solve::rust_ir;

use base_db::Crate;
use hir_def::{BlockId, TraitId, lang_item::LangItem};
use hir_expand::name::Name;
use intern::sym;
use rustc_next_trait_solver::solve::{HasChanged, SolverDelegateEvalExt};
use rustc_type_ir::{
    InferCtxtLike, TypingMode,
    inherent::{SliceLike, Span as _},
    solve::Certainty,
};
use span::Edition;
use stdx::never;
use triomphe::Arc;

use crate::{
    AliasEq, AliasTy, Canonical, DomainGoal, Goal, InEnvironment, Interner, ProjectionTy,
    ProjectionTyExt, TraitRefExt, Ty, TyKind, TypeFlags, WhereClause,
    db::HirDatabase,
    infer::unify::InferenceTable,
    next_solver::{
        DbInterner, GenericArg, SolverContext, Span,
        infer::{DbInternerInferExt, InferCtxt},
        mapping::{ChalkToNextSolver, convert_canonical_args_for_result},
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
pub struct TraitEnvironment {
    pub krate: Crate,
    pub block: Option<BlockId>,
    // FIXME make this a BTreeMap
    traits_from_clauses: Box<[(Ty, TraitId)]>,
    pub env: chalk_ir::Environment<Interner>,
}

impl TraitEnvironment {
    pub fn empty(krate: Crate) -> Arc<Self> {
        Arc::new(TraitEnvironment {
            krate,
            block: None,
            traits_from_clauses: Box::default(),
            env: chalk_ir::Environment::new(Interner),
        })
    }

    pub fn new(
        krate: Crate,
        block: Option<BlockId>,
        traits_from_clauses: Box<[(Ty, TraitId)]>,
        env: chalk_ir::Environment<Interner>,
    ) -> Arc<Self> {
        Arc::new(TraitEnvironment { krate, block, traits_from_clauses, env })
    }

    // pub fn with_block(self: &mut Arc<Self>, block: BlockId) {
    pub fn with_block(this: &mut Arc<Self>, block: BlockId) {
        Arc::make_mut(this).block = Some(block);
    }

    pub fn traits_in_scope_from_clauses(&self, ty: Ty) -> impl Iterator<Item = TraitId> + '_ {
        self.traits_from_clauses
            .iter()
            .filter_map(move |(self_ty, trait_id)| (*self_ty == ty).then_some(*trait_id))
    }
}

pub(crate) fn normalize_projection_query(
    db: &dyn HirDatabase,
    projection: ProjectionTy,
    env: Arc<TraitEnvironment>,
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

    let mut table = InferenceTable::new(db, env);
    let ty = table.normalize_projection_ty(projection);
    table.resolve_completely(ty)
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

/// Solve a trait goal using Chalk.
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

/// Solve a trait goal using Chalk.
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
        Ok((_, Certainty::Maybe(_), args)) => {
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

/// Solve a trait goal using Chalk.
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

    pub const fn to_chalk_ir(self) -> rust_ir::ClosureKind {
        // Chalk doesn't support async fn traits.
        match self {
            FnTrait::AsyncFnOnce | FnTrait::FnOnce => rust_ir::ClosureKind::FnOnce,
            FnTrait::AsyncFnMut | FnTrait::FnMut => rust_ir::ClosureKind::FnMut,
            FnTrait::AsyncFn | FnTrait::Fn => rust_ir::ClosureKind::Fn,
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

    #[inline]
    pub(crate) fn is_async(self) -> bool {
        matches!(self, FnTrait::AsyncFn | FnTrait::AsyncFnMut | FnTrait::AsyncFnOnce)
    }
}

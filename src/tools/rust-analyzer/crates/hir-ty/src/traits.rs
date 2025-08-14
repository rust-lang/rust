//! Trait solving using Chalk.

use core::fmt;
use std::env::var;

use chalk_ir::{DebruijnIndex, GoalData, fold::TypeFoldable};
use chalk_recursive::Cache;
use chalk_solve::{Solver, logging_db::LoggingRustIrDatabase, rust_ir};

use base_db::Crate;
use hir_def::{BlockId, TraitId, lang_item::LangItem};
use hir_expand::name::Name;
use intern::sym;
use span::Edition;
use stdx::{never, panic_context};
use triomphe::Arc;

use crate::{
    AliasEq, AliasTy, Canonical, DomainGoal, Goal, Guidance, InEnvironment, Interner, ProjectionTy,
    ProjectionTyExt, Solution, TraitRefExt, Ty, TyKind, TypeFlags, WhereClause, db::HirDatabase,
    infer::unify::InferenceTable, utils::UnevaluatedConstEvaluatorFolder,
};

/// This controls how much 'time' we give the Chalk solver before giving up.
const CHALK_SOLVER_FUEL: i32 = 1000;

#[derive(Debug, Copy, Clone)]
pub(crate) struct ChalkContext<'a> {
    pub(crate) db: &'a dyn HirDatabase,
    pub(crate) krate: Crate,
    pub(crate) block: Option<BlockId>,
}

fn create_chalk_solver() -> chalk_recursive::RecursiveSolver<Interner> {
    let overflow_depth =
        var("CHALK_OVERFLOW_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let max_size = var("CHALK_SOLVER_MAX_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(150);
    chalk_recursive::RecursiveSolver::new(overflow_depth, max_size, Some(Cache::new()))
}

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

/// Solve a trait goal using Chalk.
pub(crate) fn trait_solve_query(
    db: &dyn HirDatabase,
    krate: Crate,
    block: Option<BlockId>,
    goal: Canonical<InEnvironment<Goal>>,
) -> Option<Solution> {
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
        return Some(Solution::Ambig(Guidance::Unknown));
    }

    // Chalk see `UnevaluatedConst` as a unique concrete value, but we see it as an alias for another const. So
    // we should get rid of it when talking to chalk.
    let goal = goal
        .try_fold_with(&mut UnevaluatedConstEvaluatorFolder { db }, DebruijnIndex::INNERMOST)
        .unwrap();

    // We currently don't deal with universes (I think / hope they're not yet
    // relevant for our use cases?)
    let u_canonical = chalk_ir::UCanonical { canonical: goal, universes: 1 };
    solve(db, krate, block, &u_canonical)
}

fn solve(
    db: &dyn HirDatabase,
    krate: Crate,
    block: Option<BlockId>,
    goal: &chalk_ir::UCanonical<chalk_ir::InEnvironment<chalk_ir::Goal<Interner>>>,
) -> Option<chalk_solve::Solution<Interner>> {
    let _p = tracing::info_span!("solve", ?krate, ?block).entered();
    let context = ChalkContext { db, krate, block };
    tracing::debug!("solve goal: {:?}", goal);
    let mut solver = create_chalk_solver();

    let fuel = std::cell::Cell::new(CHALK_SOLVER_FUEL);

    let should_continue = || {
        db.unwind_if_revision_cancelled();
        let remaining = fuel.get();
        fuel.set(remaining - 1);
        if remaining == 0 {
            tracing::debug!("fuel exhausted");
        }
        remaining > 0
    };

    let mut solve = || {
        let _ctx = if is_chalk_debug() || is_chalk_print() {
            Some(panic_context::enter(format!("solving {goal:?}")))
        } else {
            None
        };
        let solution = if is_chalk_print() {
            let logging_db =
                LoggingRustIrDatabaseLoggingOnDrop(LoggingRustIrDatabase::new(context));
            solver.solve_limited(&logging_db.0, goal, &should_continue)
        } else {
            solver.solve_limited(&context, goal, &should_continue)
        };

        tracing::debug!("solve({:?}) => {:?}", goal, solution);

        solution
    };

    // don't set the TLS for Chalk unless Chalk debugging is active, to make
    // extra sure we only use it for debugging
    if is_chalk_debug() { crate::tls::set_current_program(db, solve) } else { solve() }
}

struct LoggingRustIrDatabaseLoggingOnDrop<'a>(LoggingRustIrDatabase<Interner, ChalkContext<'a>>);

impl Drop for LoggingRustIrDatabaseLoggingOnDrop<'_> {
    fn drop(&mut self) {
        tracing::info!("chalk program:\n{}", self.0);
    }
}

fn is_chalk_debug() -> bool {
    std::env::var("CHALK_DEBUG").is_ok()
}

fn is_chalk_print() -> bool {
    std::env::var("CHALK_PRINT").is_ok()
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

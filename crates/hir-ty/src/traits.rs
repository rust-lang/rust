//! Trait solving using Chalk.

use std::{env::var, sync::Arc};

use chalk_ir::GoalData;
use chalk_recursive::Cache;
use chalk_solve::{logging_db::LoggingRustIrDatabase, Solver};

use base_db::CrateId;
use hir_def::{
    lang_item::{LangItem, LangItemTarget},
    TraitId,
};
use stdx::panic_context;

use crate::{
    db::HirDatabase, infer::unify::InferenceTable, AliasEq, AliasTy, Canonical, DomainGoal, Goal,
    Guidance, InEnvironment, Interner, ProjectionTy, ProjectionTyExt, Solution, TraitRefExt, Ty,
    TyKind, WhereClause,
};

/// This controls how much 'time' we give the Chalk solver before giving up.
const CHALK_SOLVER_FUEL: i32 = 1000;

#[derive(Debug, Copy, Clone)]
pub(crate) struct ChalkContext<'a> {
    pub(crate) db: &'a dyn HirDatabase,
    pub(crate) krate: CrateId,
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
    pub krate: CrateId,
    // FIXME make this a BTreeMap
    pub(crate) traits_from_clauses: Vec<(Ty, TraitId)>,
    pub env: chalk_ir::Environment<Interner>,
}

impl TraitEnvironment {
    pub fn empty(krate: CrateId) -> Self {
        TraitEnvironment {
            krate,
            traits_from_clauses: Vec::new(),
            env: chalk_ir::Environment::new(Interner),
        }
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
    let mut table = InferenceTable::new(db, env);
    let ty = table.normalize_projection_ty(projection);
    table.resolve_completely(ty)
}

/// Solve a trait goal using Chalk.
pub(crate) fn trait_solve_query(
    db: &dyn HirDatabase,
    krate: CrateId,
    goal: Canonical<InEnvironment<Goal>>,
) -> Option<Solution> {
    let _p = profile::span("trait_solve_query").detail(|| match &goal.value.goal.data(Interner) {
        GoalData::DomainGoal(DomainGoal::Holds(WhereClause::Implemented(it))) => {
            db.trait_data(it.hir_trait_id()).name.to_string()
        }
        GoalData::DomainGoal(DomainGoal::Holds(WhereClause::AliasEq(_))) => "alias_eq".to_string(),
        _ => "??".to_string(),
    });
    tracing::info!("trait_solve_query({:?})", goal.value.goal);

    if let GoalData::DomainGoal(DomainGoal::Holds(WhereClause::AliasEq(AliasEq {
        alias: AliasTy::Projection(projection_ty),
        ..
    }))) = &goal.value.goal.data(Interner)
    {
        if let TyKind::BoundVar(_) = projection_ty.self_type_parameter(db).kind(Interner) {
            // Hack: don't ask Chalk to normalize with an unknown self type, it'll say that's impossible
            return Some(Solution::Ambig(Guidance::Unknown));
        }
    }

    // We currently don't deal with universes (I think / hope they're not yet
    // relevant for our use cases?)
    let u_canonical = chalk_ir::UCanonical { canonical: goal, universes: 1 };
    solve(db, krate, &u_canonical)
}

fn solve(
    db: &dyn HirDatabase,
    krate: CrateId,
    goal: &chalk_ir::UCanonical<chalk_ir::InEnvironment<chalk_ir::Goal<Interner>>>,
) -> Option<chalk_solve::Solution<Interner>> {
    let context = ChalkContext { db, krate };
    tracing::debug!("solve goal: {:?}", goal);
    let mut solver = create_chalk_solver();

    let fuel = std::cell::Cell::new(CHALK_SOLVER_FUEL);

    let should_continue = || {
        db.unwind_if_cancelled();
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
    if is_chalk_debug() {
        crate::tls::set_current_program(db, solve)
    } else {
        solve()
    }
}

struct LoggingRustIrDatabaseLoggingOnDrop<'a>(LoggingRustIrDatabase<Interner, ChalkContext<'a>>);

impl<'a> Drop for LoggingRustIrDatabaseLoggingOnDrop<'a> {
    fn drop(&mut self) {
        eprintln!("chalk program:\n{}", self.0);
    }
}

fn is_chalk_debug() -> bool {
    std::env::var("CHALK_DEBUG").is_ok()
}

fn is_chalk_print() -> bool {
    std::env::var("CHALK_PRINT").is_ok()
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FnTrait {
    FnOnce,
    FnMut,
    Fn,
}

impl FnTrait {
    const fn lang_item(self) -> LangItem {
        match self {
            FnTrait::FnOnce => LangItem::FnOnce,
            FnTrait::FnMut => LangItem::FnMut,
            FnTrait::Fn => LangItem::Fn,
        }
    }

    pub fn get_id(&self, db: &dyn HirDatabase, krate: CrateId) -> Option<TraitId> {
        let target = db.lang_item(krate, self.lang_item())?;
        match target {
            LangItemTarget::Trait(t) => Some(t),
            _ => None,
        }
    }
}

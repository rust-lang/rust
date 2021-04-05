//! Trait solving using Chalk.
use std::env::var;

use base_db::CrateId;
use chalk_ir::cast::Cast;
use chalk_solve::{logging_db::LoggingRustIrDatabase, Solver};
use hir_def::{lang_item::LangItemTarget, TraitId};
use stdx::panic_context;

use crate::{
    db::HirDatabase, AliasEq, AliasTy, Canonical, DomainGoal, Guidance, HirDisplay, InEnvironment,
    Solution, SolutionVariables, Ty, TyKind, WhereClause,
};

use self::chalk::{from_chalk, Interner, ToChalk};

pub(crate) mod chalk;

/// This controls how much 'time' we give the Chalk solver before giving up.
const CHALK_SOLVER_FUEL: i32 = 100;

#[derive(Debug, Copy, Clone)]
struct ChalkContext<'a> {
    db: &'a dyn HirDatabase,
    krate: CrateId,
}

fn create_chalk_solver() -> chalk_recursive::RecursiveSolver<Interner> {
    let overflow_depth =
        var("CHALK_OVERFLOW_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let caching_enabled = true;
    let max_size = var("CHALK_SOLVER_MAX_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    chalk_recursive::RecursiveSolver::new(overflow_depth, max_size, caching_enabled)
}

/// A set of clauses that we assume to be true. E.g. if we are inside this function:
/// ```rust
/// fn foo<T: Default>(t: T) {}
/// ```
/// we assume that `T: Default`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraitEnvironment {
    // When we're using Chalk's Ty we can make this a BTreeMap since it's Ord,
    // but for now it's too annoying...
    pub(crate) traits_from_clauses: Vec<(Ty, TraitId)>,
    pub env: chalk_ir::Environment<Interner>,
}

impl TraitEnvironment {
    pub(crate) fn traits_in_scope_from_clauses<'a>(
        &'a self,
        ty: &'a Ty,
    ) -> impl Iterator<Item = TraitId> + 'a {
        self.traits_from_clauses.iter().filter_map(move |(self_ty, trait_id)| {
            if self_ty == ty {
                Some(*trait_id)
            } else {
                None
            }
        })
    }
}

impl Default for TraitEnvironment {
    fn default() -> Self {
        TraitEnvironment {
            traits_from_clauses: Vec::new(),
            env: chalk_ir::Environment::new(&Interner),
        }
    }
}

/// Solve a trait goal using Chalk.
pub(crate) fn trait_solve_query(
    db: &dyn HirDatabase,
    krate: CrateId,
    goal: Canonical<InEnvironment<DomainGoal>>,
) -> Option<Solution> {
    let _p = profile::span("trait_solve_query").detail(|| match &goal.value.goal {
        DomainGoal::Holds(WhereClause::Implemented(it)) => {
            db.trait_data(it.hir_trait_id()).name.to_string()
        }
        DomainGoal::Holds(WhereClause::AliasEq(_)) => "alias_eq".to_string(),
    });
    log::info!("trait_solve_query({})", goal.value.goal.display(db));

    if let DomainGoal::Holds(WhereClause::AliasEq(AliasEq {
        alias: AliasTy::Projection(projection_ty),
        ..
    })) = &goal.value.goal
    {
        if let TyKind::BoundVar(_) = projection_ty.self_type_parameter(&Interner).kind(&Interner) {
            // Hack: don't ask Chalk to normalize with an unknown self type, it'll say that's impossible
            return Some(Solution::Ambig(Guidance::Unknown));
        }
    }

    let canonical = goal.to_chalk(db).cast(&Interner);

    // We currently don't deal with universes (I think / hope they're not yet
    // relevant for our use cases?)
    let u_canonical = chalk_ir::UCanonical { canonical, universes: 1 };
    let solution = solve(db, krate, &u_canonical);
    solution.map(|solution| solution_from_chalk(db, solution))
}

fn solve(
    db: &dyn HirDatabase,
    krate: CrateId,
    goal: &chalk_ir::UCanonical<chalk_ir::InEnvironment<chalk_ir::Goal<Interner>>>,
) -> Option<chalk_solve::Solution<Interner>> {
    let context = ChalkContext { db, krate };
    log::debug!("solve goal: {:?}", goal);
    let mut solver = create_chalk_solver();

    let fuel = std::cell::Cell::new(CHALK_SOLVER_FUEL);

    let should_continue = || {
        context.db.check_canceled();
        let remaining = fuel.get();
        fuel.set(remaining - 1);
        if remaining == 0 {
            log::debug!("fuel exhausted");
        }
        remaining > 0
    };

    let mut solve = || {
        let _ctx = if is_chalk_debug() || is_chalk_print() {
            Some(panic_context::enter(format!("solving {:?}", goal)))
        } else {
            None
        };
        let solution = if is_chalk_print() {
            let logging_db =
                LoggingRustIrDatabaseLoggingOnDrop(LoggingRustIrDatabase::new(context));
            let solution = solver.solve_limited(&logging_db.0, goal, &should_continue);
            solution
        } else {
            solver.solve_limited(&context, goal, &should_continue)
        };

        log::debug!("solve({:?}) => {:?}", goal, solution);

        solution
    };

    // don't set the TLS for Chalk unless Chalk debugging is active, to make
    // extra sure we only use it for debugging
    let solution =
        if is_chalk_debug() { chalk::tls::set_current_program(db, solve) } else { solve() };

    solution
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

fn solution_from_chalk(
    db: &dyn HirDatabase,
    solution: chalk_solve::Solution<Interner>,
) -> Solution {
    let convert_subst = |subst: chalk_ir::Canonical<chalk_ir::Substitution<Interner>>| {
        let result = from_chalk(db, subst);
        SolutionVariables(result)
    };
    match solution {
        chalk_solve::Solution::Unique(constr_subst) => {
            let subst = chalk_ir::Canonical {
                value: constr_subst.value.subst,
                binders: constr_subst.binders,
            };
            Solution::Unique(convert_subst(subst))
        }
        chalk_solve::Solution::Ambig(chalk_solve::Guidance::Definite(subst)) => {
            Solution::Ambig(Guidance::Definite(convert_subst(subst)))
        }
        chalk_solve::Solution::Ambig(chalk_solve::Guidance::Suggested(subst)) => {
            Solution::Ambig(Guidance::Suggested(convert_subst(subst)))
        }
        chalk_solve::Solution::Ambig(chalk_solve::Guidance::Unknown) => {
            Solution::Ambig(Guidance::Unknown)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FnTrait {
    FnOnce,
    FnMut,
    Fn,
}

impl FnTrait {
    fn lang_item_name(self) -> &'static str {
        match self {
            FnTrait::FnOnce => "fn_once",
            FnTrait::FnMut => "fn_mut",
            FnTrait::Fn => "fn",
        }
    }

    pub fn get_id(&self, db: &dyn HirDatabase, krate: CrateId) -> Option<TraitId> {
        let target = db.lang_item(krate, self.lang_item_name().into())?;
        match target {
            LangItemTarget::TraitId(t) => Some(t),
            _ => None,
        }
    }
}

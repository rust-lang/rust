//! Trait solving using Chalk.
use std::env::var;
use std::sync::Arc;

use base_db::CrateId;
use chalk_ir::cast::Cast;
use chalk_solve::{logging_db::LoggingRustIrDatabase, Solver};
use hir_def::{lang_item::LangItemTarget, TraitId};
use stdx::panic_context;

use crate::{db::HirDatabase, DebruijnIndex, Substs};

use super::{Canonical, GenericPredicate, HirDisplay, ProjectionTy, TraitRef, Ty, TypeWalk};

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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitEnvironment {
    pub predicates: Vec<GenericPredicate>,
}

impl TraitEnvironment {
    /// Returns trait refs with the given self type which are supposed to hold
    /// in this trait env. E.g. if we are in `foo<T: SomeTrait>()`, this will
    /// find that `T: SomeTrait` if we call it for `T`.
    pub(crate) fn trait_predicates_for_self_ty<'a>(
        &'a self,
        ty: &'a Ty,
    ) -> impl Iterator<Item = &'a TraitRef> + 'a {
        self.predicates.iter().filter_map(move |pred| match pred {
            GenericPredicate::Implemented(tr) if tr.self_ty() == ty => Some(tr),
            _ => None,
        })
    }
}

/// Something (usually a goal), along with an environment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InEnvironment<T> {
    pub environment: Arc<TraitEnvironment>,
    pub value: T,
}

impl<T> InEnvironment<T> {
    pub fn new(environment: Arc<TraitEnvironment>, value: T) -> InEnvironment<T> {
        InEnvironment { environment, value }
    }
}

/// Something that needs to be proven (by Chalk) during type checking, e.g. that
/// a certain type implements a certain trait. Proving the Obligation might
/// result in additional information about inference variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Obligation {
    /// Prove that a certain type implements a trait (the type is the `Self` type
    /// parameter to the `TraitRef`).
    Trait(TraitRef),
    Projection(ProjectionPredicate),
}

impl Obligation {
    pub fn from_predicate(predicate: GenericPredicate) -> Option<Obligation> {
        match predicate {
            GenericPredicate::Implemented(trait_ref) => Some(Obligation::Trait(trait_ref)),
            GenericPredicate::Projection(projection_pred) => {
                Some(Obligation::Projection(projection_pred))
            }
            GenericPredicate::Error => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProjectionPredicate {
    pub projection_ty: ProjectionTy,
    pub ty: Ty,
}

impl TypeWalk for ProjectionPredicate {
    fn walk(&self, f: &mut impl FnMut(&Ty)) {
        self.projection_ty.walk(f);
        self.ty.walk(f);
    }

    fn walk_mut_binders(
        &mut self,
        f: &mut impl FnMut(&mut Ty, DebruijnIndex),
        binders: DebruijnIndex,
    ) {
        self.projection_ty.walk_mut_binders(f, binders);
        self.ty.walk_mut_binders(f, binders);
    }
}

/// Solve a trait goal using Chalk.
pub(crate) fn trait_solve_query(
    db: &dyn HirDatabase,
    krate: CrateId,
    goal: Canonical<InEnvironment<Obligation>>,
) -> Option<Solution> {
    let _p = profile::span("trait_solve_query").detail(|| match &goal.value.value {
        Obligation::Trait(it) => db.trait_data(it.trait_).name.to_string(),
        Obligation::Projection(_) => "projection".to_string(),
    });
    log::info!("trait_solve_query({})", goal.value.value.display(db));

    if let Obligation::Projection(pred) = &goal.value.value {
        if let Ty::Bound(_) = &pred.projection_ty.parameters[0] {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolutionVariables(pub Canonical<Substs>);

#[derive(Clone, Debug, PartialEq, Eq)]
/// A (possible) solution for a proposed goal.
pub enum Solution {
    /// The goal indeed holds, and there is a unique value for all existential
    /// variables.
    Unique(SolutionVariables),

    /// The goal may be provable in multiple ways, but regardless we may have some guidance
    /// for type inference. In this case, we don't return any lifetime
    /// constraints, since we have not "committed" to any particular solution
    /// yet.
    Ambig(Guidance),
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// When a goal holds ambiguously (e.g., because there are multiple possible
/// solutions), we issue a set of *guidance* back to type inference.
pub enum Guidance {
    /// The existential variables *must* have the given values if the goal is
    /// ever to hold, but that alone isn't enough to guarantee the goal will
    /// actually hold.
    Definite(SolutionVariables),

    /// There are multiple plausible values for the existentials, but the ones
    /// here are suggested as the preferred choice heuristically. These should
    /// be used for inference fallback only.
    Suggested(SolutionVariables),

    /// There's no useful information to feed back to type inference
    Unknown,
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

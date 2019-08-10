//! Trait solving using Chalk.
use std::sync::Arc;

use chalk_ir::cast::Cast;
use log::debug;
use parking_lot::Mutex;
use ra_prof::profile;
use rustc_hash::FxHashSet;

use super::{Canonical, GenericPredicate, HirDisplay, ProjectionTy, TraitRef, Ty};
use crate::{db::HirDatabase, Crate, ImplBlock, Trait};

use self::chalk::{from_chalk, ToChalk};

pub(crate) mod chalk;

pub(crate) type Solver = chalk_solve::Solver;

/// This controls the maximum size of types Chalk considers. If we set this too
/// high, we can run into slow edge cases; if we set it too low, Chalk won't
/// find some solutions.
const CHALK_SOLVER_MAX_SIZE: usize = 4;

#[derive(Debug, Copy, Clone)]
struct ChalkContext<'a, DB> {
    db: &'a DB,
    krate: Crate,
}

pub(crate) fn trait_solver_query(_db: &impl HirDatabase, _krate: Crate) -> Arc<Mutex<Solver>> {
    // krate parameter is just so we cache a unique solver per crate
    let solver_choice = chalk_solve::SolverChoice::SLG { max_size: CHALK_SOLVER_MAX_SIZE };
    debug!("Creating new solver for crate {:?}", _krate);
    Arc::new(Mutex::new(solver_choice.into_solver()))
}

/// Collects impls for the given trait in the whole dependency tree of `krate`.
pub(crate) fn impls_for_trait_query(
    db: &impl HirDatabase,
    krate: Crate,
    trait_: Trait,
) -> Arc<[ImplBlock]> {
    let mut impls = FxHashSet::default();
    // We call the query recursively here. On the one hand, this means we can
    // reuse results from queries for different crates; on the other hand, this
    // will only ever get called for a few crates near the root of the tree (the
    // ones the user is editing), so this may actually be a waste of memory. I'm
    // doing it like this mainly for simplicity for now.
    for dep in krate.dependencies(db) {
        impls.extend(db.impls_for_trait(dep.krate, trait_).iter());
    }
    let crate_impl_blocks = db.impls_in_crate(krate);
    impls.extend(crate_impl_blocks.lookup_impl_blocks_for_trait(trait_));
    impls.into_iter().collect::<Vec<_>>().into()
}

fn solve(
    db: &impl HirDatabase,
    krate: Crate,
    goal: &chalk_ir::UCanonical<chalk_ir::InEnvironment<chalk_ir::Goal>>,
) -> Option<chalk_solve::Solution> {
    let context = ChalkContext { db, krate };
    let solver = db.trait_solver(krate);
    let solution = solver.lock().solve(&context, goal);
    debug!("solve({:?}) => {:?}", goal, solution);
    solution
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
            GenericPredicate::Error => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProjectionPredicate {
    pub projection_ty: ProjectionTy,
    pub ty: Ty,
}

/// Solve a trait goal using Chalk.
pub(crate) fn trait_solve_query(
    db: &impl HirDatabase,
    krate: Crate,
    goal: Canonical<InEnvironment<Obligation>>,
) -> Option<Solution> {
    let _p = profile("trait_solve_query");
    debug!("trait_solve_query({})", goal.value.value.display(db));
    let canonical = goal.to_chalk(db).cast();
    // We currently don't deal with universes (I think / hope they're not yet
    // relevant for our use cases?)
    let u_canonical = chalk_ir::UCanonical { canonical, universes: 1 };
    let solution = solve(db, krate, &u_canonical);
    solution.map(|solution| solution_from_chalk(db, solution))
}

fn solution_from_chalk(db: &impl HirDatabase, solution: chalk_solve::Solution) -> Solution {
    let convert_subst = |subst: chalk_ir::Canonical<chalk_ir::Substitution>| {
        let value = subst
            .value
            .parameters
            .into_iter()
            .map(|p| {
                let ty = match p {
                    chalk_ir::Parameter(chalk_ir::ParameterKind::Ty(ty)) => from_chalk(db, ty),
                    chalk_ir::Parameter(chalk_ir::ParameterKind::Lifetime(_)) => unimplemented!(),
                };
                ty
            })
            .collect();
        let result = Canonical { value, num_vars: subst.binders.len() };
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
pub struct SolutionVariables(pub Canonical<Vec<Ty>>);

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

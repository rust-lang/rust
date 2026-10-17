//! Trait Resolution. See the [rustc-dev-guide] for more information on how this works.
//!
//! [rustc-dev-guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

mod engine;
mod project;
mod structural_impls;
pub mod util;

use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::Certainty;
pub use rustc_middle::traits::*;
use rustc_middle::ty::{self, TyCtxt};
use thin_vec::ThinVec;

pub use self::engine::{FromSolverError, ScrubbedTraitError, TraitEngine, TraitErrors};
pub(crate) use self::project::UndoLog;
pub use self::project::{
    MismatchedProjectionTypes, Normalized, NormalizedTerm, ProjectionCache, ProjectionCacheEntry,
    ProjectionCacheKey, ProjectionCacheStorage,
};
use crate::infer::InferCtxt;

/// An obligation represents a predicate which must be proven in a
/// particular parameter environment.
pub type Obligation<'tcx, T> = rustc_type_ir::solve::Obligation<TyCtxt<'tcx>, T>;

pub type PredicateObligation<'tcx> = Obligation<'tcx, ty::Predicate<'tcx>>;
pub type TraitObligation<'tcx> = Obligation<'tcx, ty::TraitClause<'tcx>>;
pub type PolyTraitObligation<'tcx> = Obligation<'tcx, ty::PolyTraitClause<'tcx>>;

pub type PredicateObligations<'tcx> = ThinVec<PredicateObligation<'tcx>>;

// `PredicateObligation` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(PredicateObligation<'_>, 48);

pub type Selection<'tcx> = ImplSource<'tcx, PredicateObligation<'tcx>>;

/// A callback that can be provided to `inspect_typeck`. Invoked on evaluation
/// of root obligations.
pub type ObligationInspector<'tcx> =
    fn(&InferCtxt<'tcx>, &PredicateObligation<'tcx>, Result<Certainty, NoSolution>);

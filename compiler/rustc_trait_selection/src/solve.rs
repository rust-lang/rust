pub use rustc_next_trait_solver::solve::*;

mod delegate;
mod fulfill;
pub mod inspect;
mod normalize;
mod select;

pub(crate) use delegate::SolverDelegate;
pub use fulfill::{FulfillmentCtxt, NextSolverError};
pub(crate) use normalize::deeply_normalize_for_diagnostics;
pub use normalize::{
    deeply_normalize, deeply_normalize_with_skipped_universes,
    deeply_normalize_with_skipped_universes_and_ambiguous_goals,
};
pub use select::InferCtxtSelectExt;

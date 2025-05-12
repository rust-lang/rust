//! Experimental types for the trait query interface. The methods
//! defined in this module are all based on **canonicalization**,
//! which makes a canonical query by replacing unbound inference
//! variables and regions, so that results can be reused more broadly.
//! The providers for the queries defined here can be found in
//! `rustc_traits`.

pub mod dropck_outlives;
pub mod evaluate_obligation;
pub mod method_autoderef;
pub mod normalize;
pub mod type_op;

pub use rustc_middle::traits::query::*;

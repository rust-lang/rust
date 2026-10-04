//! Fallible versions of collections.

#![unstable(feature = "fallible_vec", issue = "157392")]

/// The error type returned by fallible collections.
// FIXME: Decide on what this error type should *really* be.
#[unstable(feature = "fallible_vec", issue = "157392")]
pub type AllocError = crate::collections::TryReserveError;

mod vec;

#[unstable(feature = "fallible_vec", issue = "157392")]
pub use vec::*;

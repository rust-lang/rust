//! Lints that run immediately after MIR building.
//!
//! By running these before any MIR transformations, we ensure that the MIR is as close to the
//! user's code as possible.

pub mod unconditional_recursion;

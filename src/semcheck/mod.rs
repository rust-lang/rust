//! This module holds all logic and data structures needed to perform semver analysis on two
//! modules which are usually crate roots (or just regular modules in a testing scenario).

pub mod changes;

mod mapping;
mod mismatch;
mod traverse;

pub use self::traverse::run_analysis;

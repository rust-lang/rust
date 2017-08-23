//! This module holds all logic and data structures needed to perform semver analysis on two
//! modules which are (usually) just crate roots.

pub mod changes;

mod mapping;
mod mismatch;
mod translate;
mod traverse;
mod typeck;

pub use self::traverse::run_analysis;

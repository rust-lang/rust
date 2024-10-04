//! Crate containing the implementation of the next-generation trait solver.
//!
//! This crate may also contain things that are used by the old trait solver,
//! but were uplifted in the process of making the new trait solver generic.
//! So if you got to this crate from the old solver, it's totally normal.

// tidy-alphabetical-start
#![cfg_attr(not(bootstrap), allow(rustc::usage_of_type_ir_inherent))]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

pub mod canonicalizer;
pub mod coherence;
pub mod delegate;
pub mod resolve;
pub mod solve;

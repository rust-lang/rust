//! The "main crate" of the Rust compiler. This crate contains common
//! type definitions that are used by the other crates in the rustc
//! "family". Some prominent examples (note that each of these modules
//! has their own README with further details).
//!
//! - **HIR.** The "high-level (H) intermediate representation (IR)" is
//!   defined in the `hir` module.
//! - **MIR.** The "mid-level (M) intermediate representation (IR)" is
//!   defined in the `mir` module. This module contains only the
//!   *definition* of the MIR; the passes that transform and operate
//!   on MIR are found in `librustc_mir` crate.
//! - **Types.** The internal representation of types used in rustc is
//!   defined in the `ty` module. This includes the **type context**
//!   (or `tcx`), which is the central context during most of
//!   compilation, containing the interners and other things.
//!
//! For more information about how rustc works, see the [rustc guide].
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]
#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(const_transmute)]
#![feature(core_intrinsics)]
#![feature(drain_filter)]
#![feature(never_type)]
#![feature(exhaustive_patterns)]
#![feature(marker_trait_attr)]
#![feature(extern_types)]
#![feature(nll)]
#![feature(option_expect_none)]
#![feature(range_is_empty)]
#![feature(specialization)]
#![feature(trusted_len)]
#![feature(vec_remove_item)]
#![feature(stmt_expr_attributes)]
#![feature(test)]
#![feature(in_band_lifetimes)]
#![feature(crate_visibility_modifier)]
#![feature(associated_type_bounds)]
#![feature(rustc_attrs)]
#![feature(hash_raw_entry)]
#![feature(int_error_matching)]
#![recursion_limit = "512"]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate scoped_tls;
#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate log;
#[macro_use]
extern crate smallvec;

#[cfg(test)]
mod tests;

#[macro_use]
mod macros;

#[macro_use]
pub mod query;

#[macro_use]
pub mod arena;
pub mod dep_graph;
pub mod hir;
pub mod ich;
pub mod infer;
pub mod lint;
pub mod middle;
pub mod mir;
pub use rustc_session as session;
pub mod traits;
pub mod ty;

pub mod util {
    pub mod bug;
    pub mod common;
}

// Allows macros to refer to this crate as `::rustc`
extern crate self as rustc;

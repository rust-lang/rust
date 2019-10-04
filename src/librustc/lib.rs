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
//! - **Traits.** Trait resolution is implemented in the `traits` module.
//! - **Type inference.** The type inference code can be found in the `infer` module;
//!   this code handles low-level equality and subtyping operations. The
//!   type check pass in the compiler is found in the `librustc_typeck` crate.
//!
//! For more information about how rustc works, see the [rustc guide].
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(arbitrary_self_types)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(const_fn)]
#![feature(const_transmute)]
#![feature(core_intrinsics)]
#![feature(drain_filter)]
#![cfg_attr(windows, feature(libc))]
#![feature(never_type)]
#![feature(exhaustive_patterns)]
#![feature(overlapping_marker_traits)]
#![feature(extern_types)]
#![feature(nll)]
#![feature(non_exhaustive)]
#![feature(optin_builtin_traits)]
#![feature(range_is_empty)]
#![feature(slice_patterns)]
#![feature(specialization)]
#![feature(unboxed_closures)]
#![feature(thread_local)]
#![feature(trace_macros)]
#![feature(trusted_len)]
#![feature(vec_remove_item)]
#![feature(stmt_expr_attributes)]
#![feature(integer_atomics)]
#![feature(test)]
#![feature(in_band_lifetimes)]
#![feature(crate_visibility_modifier)]
#![feature(proc_macro_hygiene)]
#![feature(log_syntax)]
#![feature(mem_take)]
#![feature(associated_type_bounds)]
#![feature(rustc_attrs)]

#![recursion_limit="512"]

#[macro_use] extern crate bitflags;
extern crate getopts;
#[macro_use] extern crate scoped_tls;
#[cfg(windows)]
extern crate libc;
#[macro_use] extern crate rustc_macros;
#[macro_use] extern crate rustc_data_structures;
#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
#[macro_use] extern crate smallvec;

// Use the test crate here so we depend on getopts through it. This allow tools to link to both
// librustc_driver and libtest.
extern crate test as _;

#[cfg(test)]
mod tests;

#[macro_use]
mod macros;

pub mod error_codes;

#[macro_use]
pub mod query;

#[macro_use]
pub mod arena;
pub mod dep_graph;
pub mod hir;
pub mod ich;
pub mod infer;
pub mod lint;

pub mod middle {
    pub mod expr_use_visitor;
    pub mod cstore;
    pub mod dependency_format;
    pub mod diagnostic_items;
    pub mod exported_symbols;
    pub mod free_region;
    pub mod lib_features;
    pub mod lang_items;
    pub mod mem_categorization;
    pub mod privacy;
    pub mod reachable;
    pub mod region;
    pub mod recursion_limit;
    pub mod resolve_lifetime;
    pub mod stability;
    pub mod weak_lang_items;
}

pub mod mir;
pub mod session;
pub mod traits;
pub mod ty;

pub mod util {
    pub mod captures;
    pub mod common;
    pub mod nodemap;
    pub mod profiling;
    pub mod bug;
}

// Allows macros to refer to this crate as `::rustc`
extern crate self as rustc;

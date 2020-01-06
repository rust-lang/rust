//! This crates defines the trait resolution method and the type inference engine.
//!
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
#![feature(bool_to_option)]
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
#![feature(optin_builtin_traits)]
#![feature(option_expect_none)]
#![feature(range_is_empty)]
#![feature(specialization)]
#![feature(unboxed_closures)]
#![feature(thread_local)]
#![feature(trace_macros)]
#![feature(trusted_len)]
#![feature(stmt_expr_attributes)]
#![feature(integer_atomics)]
#![feature(test)]
#![feature(in_band_lifetimes)]
#![feature(crate_visibility_modifier)]
#![feature(log_syntax)]
#![feature(associated_type_bounds)]
#![feature(rustc_attrs)]
#![feature(hash_raw_entry)]
#![recursion_limit = "512"]

#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate log;
#[macro_use]
extern crate rustc;

use rustc::arena;
use rustc::dep_graph;
use rustc::hir;
pub mod infer;
use rustc::middle;
pub use rustc_session as session;
pub mod traits;
use rustc::ty;
use rustc::util;

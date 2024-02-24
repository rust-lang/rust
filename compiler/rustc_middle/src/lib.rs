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
//!   on MIR are found in `rustc_const_eval` crate.
//! - **Types.** The internal representation of types used in rustc is
//!   defined in the `ty` module. This includes the **type context**
//!   (or `tcx`), which is the central context during most of
//!   compilation, containing the interners and other things.
//!
//! For more information about how rustc works, see the [rustc dev guide].
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![cfg_attr(bootstrap, feature(exhaustive_patterns))]
#![cfg_attr(not(bootstrap), feature(min_exhaustive_patterns))]
#![feature(rustdoc_internals)]
#![feature(allocator_api)]
#![feature(array_windows)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(core_intrinsics)]
#![feature(const_type_name)]
#![feature(discriminant_kind)]
#![feature(coroutines)]
#![feature(generic_nonzero)]
#![feature(if_let_guard)]
#![feature(inline_const)]
#![feature(iter_from_coroutine)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(extern_types)]
#![feature(new_uninit)]
#![feature(let_chains)]
#![feature(min_specialization)]
#![feature(trusted_len)]
#![feature(type_alias_impl_trait)]
#![feature(strict_provenance)]
#![feature(associated_type_bounds)]
#![feature(rustc_attrs)]
#![feature(control_flow_enum)]
#![feature(trait_upcasting)]
#![feature(try_blocks)]
#![feature(decl_macro)]
#![feature(extract_if)]
#![feature(intra_doc_pointers)]
#![feature(yeet_expr)]
#![feature(const_option)]
#![feature(ptr_alignment_type)]
#![feature(macro_metavar_expr)]
#![allow(internal_features)]
#![allow(rustc::potential_query_instability)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate tracing;
#[macro_use]
extern crate smallvec;

#[cfg(test)]
mod tests;

#[macro_use]
mod macros;

#[macro_use]
pub mod arena;
pub mod error;
pub mod hir;
pub mod hooks;
pub mod infer;
pub mod lint;
pub mod metadata;
pub mod middle;
pub mod mir;
pub mod thir;
pub mod traits;
pub mod ty;
pub mod util;
mod values;

#[macro_use]
pub mod query;
#[macro_use]
pub mod dep_graph;

// Allows macros to refer to this crate as `::rustc_middle`
extern crate self as rustc_middle;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

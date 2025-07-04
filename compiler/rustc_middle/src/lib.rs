//! The "main crate" of the Rust compiler. This crate contains common
//! type definitions that are used by the other crates in the rustc
//! "family". The following are some prominent examples.
//!
//! - **HIR.** The "high-level (H) intermediate representation (IR)" is
//!   defined in the [`hir`] module.
//! - **THIR.** The "typed high-level (H) intermediate representation (IR)"
//!   is defined in the [`thir`] module.
//! - **MIR.** The "mid-level (M) intermediate representation (IR)" is
//!   defined in the [`mir`] module. This module contains only the
//!   *definition* of the MIR; the passes that transform and operate
//!   on MIR are found in `rustc_const_eval` crate.
//! - **Types.** The internal representation of types used in rustc is
//!   defined in the [`ty`] module. This includes the
//!   [**type context**][ty::TyCtxt] (or `tcx`), which is the central
//!   context during most of compilation, containing the interners and
//!   other things.
//!
//! For more information about how rustc works, see the [rustc dev guide].
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::direct_use_of_rustc_type_ir)]
#![allow(rustc::untranslatable_diagnostic)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(allocator_api)]
#![feature(array_windows)]
#![feature(assert_matches)]
#![feature(associated_type_defaults)]
#![feature(box_as_ptr)]
#![feature(box_patterns)]
#![feature(closure_track_caller)]
#![feature(core_intrinsics)]
#![feature(debug_closure_helpers)]
#![feature(decl_macro)]
#![feature(discriminant_kind)]
#![feature(extern_types)]
#![feature(file_buffered)]
#![feature(gen_blocks)]
#![feature(if_let_guard)]
#![feature(intra_doc_pointers)]
#![feature(min_specialization)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(ptr_alignment_type)]
#![feature(round_char_boundary)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(sized_hierarchy)]
#![feature(try_blocks)]
#![feature(try_trait_v2)]
#![feature(try_trait_v2_yeet)]
#![feature(type_alias_impl_trait)]
#![feature(yeet_expr)]
#![recursion_limit = "256"]
// tidy-alphabetical-end

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

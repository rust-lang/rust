//! This crate defines the trait resolution method.
//!
//! - **Traits.** Trait resolution is implemented in the `traits` module.
//!
//! For more information about how rustc works, see the [rustc-dev-guide].
//!
//! [rustc-dev-guide]: https://rustc-dev-guide.rust-lang.org/
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(associated_type_defaults)]
#![feature(box_patterns)]
#![feature(control_flow_enum)]
#![feature(extract_if)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(rustdoc_internals)]
#![feature(type_alias_impl_trait)]
#![recursion_limit = "512"] // For rustdoc
// tidy-alphabetical-end

#[macro_use]
extern crate tracing;

pub mod errors;
pub mod infer;
pub mod regions;
pub mod solve;
pub mod traits;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

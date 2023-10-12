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

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![cfg_attr(not(bootstrap), doc(rust_logo))]
#![cfg_attr(not(bootstrap), feature(rustdoc_internals))]
#![cfg_attr(not(bootstrap), allow(internal_features))]
#![feature(associated_type_bounds)]
#![feature(box_patterns)]
#![feature(control_flow_enum)]
#![feature(extract_if)]
#![feature(let_chains)]
#![feature(if_let_guard)]
#![feature(never_type)]
#![feature(result_option_inspect)]
#![feature(type_alias_impl_trait)]
#![feature(min_specialization)]
#![recursion_limit = "512"] // For rustdoc

#[macro_use]
extern crate rustc_macros;
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate smallvec;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;

pub mod errors;
pub mod infer;
pub mod solve;
pub mod traits;

fluent_messages! { "../messages.ftl" }

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

#![allow(rustc::potential_query_instability)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(box_patterns)]
#![feature(control_flow_enum)]
#![feature(drain_filter)]
#![feature(hash_drain_filter)]
#![cfg_attr(bootstrap, feature(label_break_value))]
#![feature(let_else)]
#![feature(if_let_guard)]
#![feature(never_type)]
#![feature(type_alias_impl_trait)]
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

pub mod autoderef;
pub mod infer;
pub mod traits;

//! This crates defines the type inference engine.
//!
//! - **Type inference.** The type inference code can be found in the `infer` module;
//!   this code handles low-level equality and subtyping operations. The
//!   type check pass in the compiler is found in the `rustc_typeck` crate.
//!
//! For more information about how rustc works, see the [rustc dev guide].
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![allow(rustc::potential_query_instability)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(box_patterns)]
#![feature(control_flow_enum)]
#![feature(extend_one)]
#![cfg_attr(bootstrap, feature(label_break_value))]
#![feature(let_else)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(try_blocks)]
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

pub mod infer;
pub mod traits;

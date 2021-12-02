//! This crates defines the trait resolution method.
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
#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(drain_filter)]
#![feature(derive_default_enum)]
#![feature(hash_drain_filter)]
#![feature(in_band_lifetimes)]
#![feature(iter_zip)]
#![feature(let_else)]
#![feature(never_type)]
#![feature(crate_visibility_modifier)]
#![feature(control_flow_enum)]
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
pub mod opaque_types;
pub mod traits;

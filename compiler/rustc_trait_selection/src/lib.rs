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

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]
#![feature(bool_to_option)]
#![feature(drain_filter)]
#![feature(in_band_lifetimes)]
#![feature(crate_visibility_modifier)]
#![feature(or_patterns)]
#![recursion_limit = "512"] // For rustdoc

#[macro_use]
extern crate rustc_macros;
#[cfg(target_arch = "x86_64")]
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

pub mod autoderef;
pub mod infer;
pub mod opaque_types;
pub mod traits;

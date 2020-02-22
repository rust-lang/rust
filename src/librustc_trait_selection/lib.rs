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
#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(drain_filter)]
#![feature(never_type)]
#![feature(range_is_empty)]
#![feature(in_band_lifetimes)]
#![feature(crate_visibility_modifier)]
#![recursion_limit = "512"]

#[macro_use]
extern crate rustc_macros;
#[cfg(target_arch = "x86_64")]
#[macro_use]
extern crate rustc_data_structures;
#[macro_use]
extern crate log;
#[macro_use]
extern crate rustc;

pub mod infer;
pub mod opaque_types;
pub mod traits;

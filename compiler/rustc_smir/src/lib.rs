//! The WIP stable interface to rustc internals.
//!
//! For more information see <https://github.com/rust-lang/project-stable-mir>
//!
//! # Note
//!
//! This API is still completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::usage_of_ty_tykind)]
#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

pub mod rustc_internal;

pub mod rustc_smir;

pub mod stable_mir;

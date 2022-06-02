//! The WIP stable interface to rustc internals.
//!
//! For more information see https://github.com/rust-lang/project-stable-mir
//!
//! # Note
//!
//! This API is still completely unstable and subject to change.

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![cfg_attr(not(feature = "default"), feature(rustc_private))]

pub mod mir;

pub mod very_unstable;

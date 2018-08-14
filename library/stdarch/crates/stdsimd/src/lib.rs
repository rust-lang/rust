//! SIMD and vendor intrinsics support library.
//!
//! This crate defines the vendor intrinsics and types primarily used for SIMD
//! in Rust.

#![feature(const_fn, integer_atomics, staged_api, stdsimd)]
#![feature(doc_cfg, allow_internal_unstable)]
#![cfg_attr(feature = "cargo-clippy", allow(shadow_reuse))]
#![cfg_attr(target_os = "linux", feature(linkage))]
#![no_std]
#![unstable(feature = "stdsimd", issue = "27731")]

#[macro_use]
extern crate cfg_if;
extern crate coresimd;
extern crate libc;
extern crate std as __do_not_use_this_import;

#[cfg(test)]
#[macro_use(println, print)]
extern crate std;

#[path = "../../../stdsimd/mod.rs"]
mod stdsimd;

pub use stdsimd::*;

#[allow(unused_imports)]
use __do_not_use_this_import::fs;
#[allow(unused_imports)]
use __do_not_use_this_import::io;
#[allow(unused_imports)]
use __do_not_use_this_import::mem;
#[allow(unused_imports)]
use __do_not_use_this_import::prelude;

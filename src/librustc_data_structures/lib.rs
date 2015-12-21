// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Various data structures used by the Rust compiler. The intention
//! is that code in here should be not be *specific* to rustc, so that
//! it can be easily unit tested and so forth.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![crate_name = "rustc_data_structures"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://www.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(rustc_private, staged_api)]
#![feature(hashmap_hasher)]

#![cfg_attr(test, feature(test))]

#[macro_use] extern crate log;
extern crate serialize as rustc_serialize; // used by deriving

pub mod bitvec;
pub mod graph;
pub mod ivar;
pub mod snapshot_vec;
pub mod transitive_relation;
pub mod unify;
pub mod fnv;
pub mod tuple_slice;

// See comments in src/librustc/lib.rs
#[doc(hidden)]
pub fn __noop_fix_for_27438() {}

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

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://www.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(shared)]
#![feature(collections_range)]
#![feature(nonzero)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(unsize)]
#![feature(i128_type)]
#![feature(i128)]
#![feature(conservative_impl_trait)]
#![feature(specialization)]
#![feature(optin_builtin_traits)]
#![feature(underscore_lifetimes)]
#![feature(macro_vis_matcher)]
#![feature(allow_internal_unstable)]

#![cfg_attr(unix, feature(libc))]
#![cfg_attr(test, feature(test))]

extern crate core;
#[macro_use]
extern crate log;
extern crate serialize as rustc_serialize; // used by deriving
#[cfg(unix)]
extern crate libc;
extern crate parking_lot;
#[macro_use]
extern crate cfg_if;
extern crate stable_deref_trait;

pub use rustc_serialize::hex::ToHex;

pub mod array_vec;
pub mod accumulate_vec;
pub mod small_vec;
pub mod base_n;
pub mod bitslice;
pub mod bitvec;
pub mod blake2b;
pub mod graph;
pub mod indexed_set;
pub mod indexed_vec;
pub mod obligation_forest;
pub mod sip128;
pub mod snapshot_map;
pub mod snapshot_vec;
pub mod stable_hasher;
pub mod transitive_relation;
pub mod unify;
pub mod fx;
pub mod tuple_slice;
pub mod veccell;
pub mod control_flow_graph;
pub mod flock;
pub mod sync;
pub mod owning_ref;

// See comments in src/librustc/lib.rs
#[doc(hidden)]
pub fn __noop_fix_for_27438() {}

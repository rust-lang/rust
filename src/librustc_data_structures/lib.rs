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

#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(unsize)]
#![feature(specialization)]
#![feature(optin_builtin_traits)]
#![cfg_attr(stage0, feature(macro_vis_matcher))]
#![cfg_attr(not(stage0), feature(nll))]
#![feature(allow_internal_unstable)]
#![feature(vec_resize_with)]

#![cfg_attr(unix, feature(libc))]
#![cfg_attr(test, feature(test))]

extern crate core;
extern crate ena;
#[macro_use]
extern crate log;
extern crate serialize as rustc_serialize; // used by deriving
#[cfg(unix)]
extern crate libc;
extern crate parking_lot;
#[macro_use]
extern crate cfg_if;
extern crate stable_deref_trait;
extern crate rustc_rayon as rayon;
extern crate rustc_rayon_core as rayon_core;
extern crate rustc_hash;
extern crate serialize;
#[cfg_attr(test, macro_use)]
extern crate smallvec;

// See librustc_cratesio_shim/Cargo.toml for a comment explaining this.
#[allow(unused_extern_crates)]
extern crate rustc_cratesio_shim;

pub use rustc_serialize::hex::ToHex;

pub mod svh;
pub mod accumulate_vec;
pub mod array_vec;
pub mod base_n;
pub mod bitslice;
pub mod bitvec;
pub mod const_cstr;
pub mod flock;
pub mod fx;
pub mod graph;
pub mod indexed_set;
pub mod indexed_vec;
pub mod obligation_forest;
pub mod owning_ref;
pub mod ptr_key;
pub mod sip128;
pub mod small_c_str;
pub mod small_vec;
pub mod snapshot_map;
pub use ena::snapshot_vec;
pub mod sorted_map;
#[macro_use] pub mod stable_hasher;
pub mod sync;
pub mod tiny_list;
pub mod thin_vec;
pub mod transitive_relation;
pub mod tuple_slice;
pub use ena::unify;
pub mod work_queue;
pub mod fingerprint;

pub struct OnDrop<F: Fn()>(pub F);

impl<F: Fn()> OnDrop<F> {
      /// Forgets the function which prevents it from running.
      /// Ensure that the function owns no memory, otherwise it will be leaked.
      pub fn disable(self) {
            std::mem::forget(self);
      }
}

impl<F: Fn()> Drop for OnDrop<F> {
      fn drop(&mut self) {
            (self.0)();
      }
}

// See comments in src/librustc/lib.rs
#[doc(hidden)]
pub fn __noop_fix_for_27438() {}

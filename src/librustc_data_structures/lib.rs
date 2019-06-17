//! Various data structures used by the Rust compiler. The intention
//! is that code in here should be not be *specific* to rustc, so that
//! it can be easily unit tested and so forth.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(in_band_lifetimes)]
#![feature(unboxed_closures)]
#![feature(generators)]
#![feature(generator_trait)]
#![feature(fn_traits)]
#![feature(unsize)]
#![feature(specialization)]
#![feature(optin_builtin_traits)]
#![feature(nll)]
#![feature(allow_internal_unstable)]
#![feature(hash_raw_entry)]
#![feature(stmt_expr_attributes)]
#![feature(core_intrinsics)]
#![feature(integer_atomics)]

#![cfg_attr(unix, feature(libc))]
#![cfg_attr(test, feature(test))]

#![deny(rust_2018_idioms)]

#[macro_use]
extern crate log;
#[allow(unused_extern_crates)]
extern crate serialize as rustc_serialize; // used by deriving
#[cfg(unix)]
extern crate libc;
#[macro_use]
extern crate cfg_if;

// See librustc_cratesio_shim/Cargo.toml for a comment explaining this.
#[allow(unused_extern_crates)]
extern crate rustc_cratesio_shim;

pub use rustc_serialize::hex::ToHex;

#[inline(never)]
#[cold]
pub fn cold_path<F: FnOnce() -> R, R>(f: F) -> R {
      f()
}

#[macro_export]
macro_rules! likely {
      ($e:expr) => {
            #[allow(unused_unsafe)]
            {
                  unsafe { std::intrinsics::likely($e) }
            }
      }
}

#[macro_export]
macro_rules! unlikely {
    ($e:expr) => {
            #[allow(unused_unsafe)]
            {
                  unsafe { std::intrinsics::unlikely($e) }
            }
      }
}

pub mod macros;
pub mod svh;
pub mod base_n;
pub mod binary_search_util;
pub mod bit_set;
pub mod box_region;
pub mod const_cstr;
pub mod flock;
pub mod fx;
pub mod graph;
pub mod indexed_vec;
pub mod interner;
pub mod jobserver;
pub mod obligation_forest;
pub mod owning_ref;
pub mod ptr_key;
pub mod sip128;
pub mod small_c_str;
pub mod snapshot_map;
pub use ena::snapshot_vec;
pub mod sorted_map;
#[macro_use] pub mod stable_hasher;
pub mod sync;
pub mod tiny_list;
pub mod thin_vec;
pub mod transitive_relation;
pub use ena::unify;
pub mod vec_linked_list;
pub mod work_queue;
pub mod fingerprint;

pub struct OnDrop<F: Fn()>(pub F);

impl<F: Fn()> OnDrop<F> {
      /// Forgets the function which prevents it from running.
      /// Ensure that the function owns no memory, otherwise it will be leaked.
      #[inline]
      pub fn disable(self) {
            std::mem::forget(self);
      }
}

impl<F: Fn()> Drop for OnDrop<F> {
      #[inline]
      fn drop(&mut self) {
            (self.0)();
      }
}

// See comments in src/librustc/lib.rs
#[doc(hidden)]
pub fn __noop_fix_for_27438() {}

//! Various data structures used by the Rust compiler. The intention
//! is that code in here should not be *specific* to rustc, so that
//! it can be easily unit tested and so forth.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::default_hash_types)]
#![allow(rustc::potential_query_instability)]
#![cfg_attr(bootstrap, feature(cfg_match))]
#![cfg_attr(not(bootstrap), feature(cfg_select))]
#![deny(unsafe_op_in_unsafe_fn)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(allocator_api)]
#![feature(array_windows)]
#![feature(ascii_char)]
#![feature(ascii_char_variants)]
#![feature(assert_matches)]
#![feature(auto_traits)]
#![feature(core_intrinsics)]
#![feature(dropck_eyepatch)]
#![feature(extend_one)]
#![feature(file_buffered)]
#![feature(map_try_insert)]
#![feature(min_specialization)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(ptr_alignment_type)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(test)]
#![feature(thread_id_value)]
#![feature(type_alias_impl_trait)]
#![feature(unwrap_infallible)]
// tidy-alphabetical-end

use std::fmt;

pub use atomic_ref::AtomicRef;
pub use ena::{snapshot_vec, undo_log, unify};
pub use rustc_index::static_assert_size;

pub mod aligned;
pub mod base_n;
pub mod binary_search_util;
pub mod fingerprint;
pub mod flat_map_in_place;
pub mod flock;
pub mod frozen;
pub mod fx;
pub mod graph;
pub mod intern;
pub mod jobserver;
pub mod marker;
pub mod memmap;
pub mod obligation_forest;
pub mod owned_slice;
pub mod packed;
pub mod profiling;
pub mod sharded;
pub mod small_c_str;
pub mod snapshot_map;
pub mod sorted_map;
pub mod sso;
pub mod stable_hasher;
pub mod stack;
pub mod steal;
pub mod svh;
pub mod sync;
pub mod tagged_ptr;
pub mod temp_dir;
pub mod thinvec;
pub mod thousands;
pub mod transitive_relation;
pub mod unhash;
pub mod unord;
pub mod vec_cache;
pub mod work_queue;

mod atomic_ref;

/// This calls the passed function while ensuring it won't be inlined into the caller.
#[inline(never)]
#[cold]
pub fn outline<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

/// Returns a structure that calls `f` when dropped.
pub fn defer<F: FnOnce()>(f: F) -> OnDrop<F> {
    OnDrop(Some(f))
}

pub struct OnDrop<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> OnDrop<F> {
    /// Disables on-drop call.
    #[inline]
    pub fn disable(mut self) {
        self.0.take();
    }
}

impl<F: FnOnce()> Drop for OnDrop<F> {
    #[inline]
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

/// This is a marker for a fatal compiler error used with `resume_unwind`.
pub struct FatalErrorMarker;

/// Turns a closure that takes an `&mut Formatter` into something that can be display-formatted.
pub fn make_display(f: impl Fn(&mut fmt::Formatter<'_>) -> fmt::Result) -> impl fmt::Display {
    struct Printer<F> {
        f: F,
    }
    impl<F> fmt::Display for Printer<F>
    where
        F: Fn(&mut fmt::Formatter<'_>) -> fmt::Result,
    {
        fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
            (self.f)(fmt)
        }
    }

    Printer { f }
}

// See comment in compiler/rustc_middle/src/tests.rs and issue #27438.
#[doc(hidden)]
pub fn __noop_fix_for_windows_dllimport_issue() {}

#[macro_export]
macro_rules! external_bitflags_debug {
    ($Name:ident) => {
        impl ::std::fmt::Debug for $Name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                ::bitflags::parser::to_writer(self, f)
            }
        }
    };
}

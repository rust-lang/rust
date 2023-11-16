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
#![deny(rustc::diagnostic_outside_of_impl)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(unsafe_op_in_unsafe_fn)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(allocator_api)]
#![feature(array_windows)]
#![feature(auto_traits)]
#![feature(cell_leak)]
#![feature(cfg_match)]
#![feature(core_intrinsics)]
#![feature(extend_one)]
#![feature(hash_raw_entry)]
#![feature(hasher_prefixfree_extras)]
#![feature(lazy_cell)]
#![feature(lint_reasons)]
#![feature(macro_metavar_expr)]
#![feature(maybe_uninit_uninit_array)]
#![feature(min_specialization)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(ptr_alignment_type)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(strict_provenance)]
#![feature(test)]
#![feature(thread_id_value)]
#![feature(type_alias_impl_trait)]
#![feature(unwrap_infallible)]
// tidy-alphabetical-end

#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_macros;

use std::fmt;

pub use rustc_index::static_assert_size;

/// This calls the passed function while ensuring it won't be inlined into the caller.
#[inline(never)]
#[cold]
pub fn outline<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

pub mod base_n;
pub mod binary_search_util;
pub mod captures;
pub mod flat_map_in_place;
pub mod flock;
pub mod fx;
pub mod graph;
pub mod intern;
pub mod jobserver;
pub mod macros;
pub mod obligation_forest;
pub mod sip128;
pub mod small_c_str;
pub mod snapshot_map;
pub mod svh;
pub use ena::snapshot_vec;
pub mod memmap;
pub mod sorted_map;
#[macro_use]
pub mod stable_hasher;
mod atomic_ref;
pub mod fingerprint;
pub mod marker;
pub mod profiling;
pub mod sharded;
pub mod stack;
pub mod sync;
pub mod tiny_list;
pub mod transitive_relation;
pub mod vec_linked_list;
pub mod work_queue;
pub use atomic_ref::AtomicRef;
pub mod aligned;
pub mod frozen;
mod hashes;
pub mod owned_slice;
pub mod sso;
pub mod steal;
pub mod tagged_ptr;
pub mod temp_dir;
pub mod unhash;
pub mod unord;

pub use ena::undo_log;
pub use ena::unify;

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

// See comments in src/librustc_middle/lib.rs
#[doc(hidden)]
pub fn __noop_fix_for_27438() {}

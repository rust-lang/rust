#![cfg(test)]
#![allow(unused_attributes)]
#![unstable(feature = "alloctests", issue = "none")]
#![no_std]
// Lints:
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(deprecated_in_future)]
#![warn(missing_debug_implementations)]
#![allow(explicit_outlives_requirements)]
#![allow(internal_features)]
#![allow(rustdoc::redundant_explicit_links)]
#![warn(rustdoc::unescaped_backticks)]
#![deny(ffi_unwind_calls)]
//
// Library features:
// tidy-alphabetical-start
#![feature(alloc_layout_extra)]
#![feature(allocator_api)]
#![feature(array_into_iter_constructors)]
#![feature(assert_matches)]
#![feature(core_intrinsics)]
#![feature(exact_size_is_empty)]
#![feature(extend_one)]
#![feature(extend_one_unchecked)]
#![feature(hasher_prefixfree_extras)]
#![feature(inplace_iteration)]
#![feature(iter_advance_by)]
#![feature(iter_next_chunk)]
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_uninit_array_transpose)]
#![feature(ptr_alignment_type)]
#![feature(ptr_internals)]
#![feature(sized_type_properties)]
#![feature(slice_iter_mut_as_mut_slice)]
#![feature(slice_ptr_get)]
#![feature(slice_range)]
#![feature(std_internals)]
#![feature(temporary_niche_types)]
#![feature(trusted_fused)]
#![feature(trusted_len)]
#![feature(trusted_random_access)]
#![feature(try_reserve_kind)]
#![feature(try_trait_v2)]
// tidy-alphabetical-end
//
// Language features:
// tidy-alphabetical-start
#![feature(cfg_sanitize)]
#![feature(dropck_eyepatch)]
#![feature(lang_items)]
#![feature(min_specialization)]
#![feature(negative_impls)]
#![feature(never_type)]
#![feature(optimize_attribute)]
#![feature(rustc_allow_const_fn_unstable)]
#![feature(rustc_attrs)]
#![feature(staged_api)]
#![feature(test)]
#![rustc_preserve_ub_checks]
// tidy-alphabetical-end

// Allow testing this library
extern crate alloc as realalloc;
#[macro_use]
extern crate std;
#[cfg(test)]
extern crate test;
mod testing;
use realalloc::*;

// We are directly including collections and raw_vec here as both use non-public
// methods and fields in tests and as such need to have the types to test in the
// same crate as the tests themself.
#[path = "../alloc/src/collections/mod.rs"]
mod collections;

#[path = "../alloc/src/raw_vec/mod.rs"]
mod raw_vec;

#[allow(dead_code)] // Not used in all configurations
pub(crate) mod test_helpers {
    /// Copied from `std::test_helpers::test_rng`, since these tests rely on the
    /// seed not being the same for every RNG invocation too.
    pub(crate) fn test_rng() -> rand_xorshift::XorShiftRng {
        use std::hash::{BuildHasher, Hash, Hasher};
        let mut hasher = std::hash::RandomState::new().build_hasher();
        std::panic::Location::caller().hash(&mut hasher);
        let hc64 = hasher.finish();
        let seed_vec =
            hc64.to_le_bytes().into_iter().chain(0u8..8).collect::<crate::vec::Vec<u8>>();
        let seed: [u8; 16] = seed_vec.as_slice().try_into().unwrap();
        rand::SeedableRng::from_seed(seed)
    }
}

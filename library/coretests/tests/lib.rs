// tidy-alphabetical-start
#![cfg_attr(target_has_atomic = "128", feature(integer_atomics))]
#![cfg_attr(test, feature(cfg_match))]
#![feature(alloc_layout_extra)]
#![feature(array_chunks)]
#![feature(array_ptr_get)]
#![feature(array_try_from_fn)]
#![feature(array_windows)]
#![feature(ascii_char)]
#![feature(ascii_char_variants)]
#![feature(async_iter_from_iter)]
#![feature(async_iterator)]
#![feature(bigint_helper_methods)]
#![feature(bstr)]
#![feature(char_max_len)]
#![feature(clone_to_uninit)]
#![feature(const_eval_select)]
#![feature(const_swap_nonoverlapping)]
#![feature(const_trait_impl)]
#![feature(core_intrinsics)]
#![feature(core_intrinsics_fallbacks)]
#![feature(core_io_borrowed_buf)]
#![feature(core_private_bignum)]
#![feature(core_private_diy_float)]
#![feature(dec2flt)]
#![feature(duration_constants)]
#![feature(duration_constructors)]
#![feature(error_generic_member_access)]
#![feature(exact_size_is_empty)]
#![feature(extend_one)]
#![feature(extern_types)]
#![feature(float_minimum_maximum)]
#![feature(flt2dec)]
#![feature(fmt_internals)]
#![feature(formatting_options)]
#![feature(freeze)]
#![feature(future_join)]
#![feature(generic_assert_internals)]
#![feature(hasher_prefixfree_extras)]
#![feature(hashmap_internals)]
#![feature(int_roundings)]
#![feature(ip)]
#![feature(ip_from)]
#![feature(is_ascii_octdigit)]
#![feature(isolate_most_least_significant_one)]
#![feature(iter_advance_by)]
#![feature(iter_array_chunks)]
#![feature(iter_chain)]
#![feature(iter_collect_into)]
#![feature(iter_intersperse)]
#![feature(iter_is_partitioned)]
#![feature(iter_map_windows)]
#![feature(iter_next_chunk)]
#![feature(iter_order_by)]
#![feature(iter_partition_in_place)]
#![feature(iterator_try_collect)]
#![feature(iterator_try_reduce)]
#![feature(layout_for_ptr)]
#![feature(lazy_get)]
#![feature(maybe_uninit_fill)]
#![feature(maybe_uninit_uninit_array_transpose)]
#![feature(maybe_uninit_write_slice)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(next_index)]
#![feature(numfmt)]
#![feature(pattern)]
#![feature(pointer_is_aligned_to)]
#![feature(portable_simd)]
#![feature(ptr_metadata)]
#![feature(select_unpredictable)]
#![feature(slice_from_ptr_range)]
#![feature(slice_internals)]
#![feature(slice_partition_dedup)]
#![feature(slice_split_once)]
#![feature(split_array)]
#![feature(split_as_slice)]
#![feature(std_internals)]
#![feature(step_trait)]
#![feature(str_internals)]
#![feature(strict_provenance_atomic_ptr)]
#![feature(strict_provenance_lints)]
#![feature(test)]
#![feature(trusted_len)]
#![feature(trusted_random_access)]
#![feature(try_blocks)]
#![feature(try_find)]
#![feature(try_trait_v2)]
#![feature(unsize)]
#![feature(unwrap_infallible)]
// tidy-alphabetical-end
#![allow(internal_features)]
#![deny(fuzzy_provenance_casts)]
#![deny(unsafe_op_in_unsafe_fn)]

/// Version of `assert_matches` that ignores fancy runtime printing in const context and uses structural equality.
macro_rules! assert_eq_const_safe {
    ($t:ty: $left:expr, $right:expr) => {
        assert_eq_const_safe!($t: $left, $right, concat!(stringify!($left), " == ", stringify!($right)));
    };
    ($t:ty: $left:expr, $right:expr$(, $($arg:tt)+)?) => {
        {
            fn runtime() {
                assert_eq!($left, $right, $($($arg)*),*);
            }
            const fn compiletime() {
                const PAT: $t = $right;
                assert!(matches!($left, PAT), $($($arg)*),*);
            }
            core::intrinsics::const_eval_select((), compiletime, runtime)
        }
    };
}

/// Creates a test for runtime and a test for constant-time.
macro_rules! test_runtime_and_compiletime {
    ($(
        $(#[$attr:meta])*
        fn $test:ident() $block:block
    )*) => {
        $(
            $(#[$attr])*
            #[test]
            fn $test() $block
            $(#[$attr])*
            const _: () = $block;
        )*
    }
}

mod alloc;
mod any;
mod array;
mod ascii;
mod ascii_char;
mod asserting;
mod async_iter;
mod atomic;
mod bool;
mod bstr;
mod cell;
mod char;
mod clone;
mod cmp;
mod const_ptr;
mod convert;
mod ffi;
mod fmt;
mod future;
mod hash;
mod hint;
mod intrinsics;
mod io;
mod iter;
mod lazy;
mod macros;
mod manually_drop;
mod mem;
mod net;
mod nonzero;
mod num;
mod ops;
mod option;
mod panic;
mod pattern;
mod pin;
mod pin_macro;
mod ptr;
mod result;
mod simd;
mod slice;
mod str;
mod str_lossy;
mod task;
mod time;
mod tuple;
mod unicode;
mod waker;

/// Copied from `std::test_helpers::test_rng`, see that function for rationale.
#[track_caller]
#[allow(dead_code)] // Not used in all configurations.
pub(crate) fn test_rng() -> rand_xorshift::XorShiftRng {
    use core::hash::{BuildHasher, Hash, Hasher};
    let mut hasher = std::hash::RandomState::new().build_hasher();
    core::panic::Location::caller().hash(&mut hasher);
    let hc64 = hasher.finish();
    let seed_vec = hc64.to_le_bytes().into_iter().chain(0u8..8).collect::<Vec<u8>>();
    let seed: [u8; 16] = seed_vec.as_slice().try_into().unwrap();
    rand::SeedableRng::from_seed(seed)
}

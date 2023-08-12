// tidy-alphabetical-start
#![deny(fuzzy_provenance_casts)]
#![deny(unsafe_op_in_unsafe_fn)]
#![feature(alloc_layout_extra)]
#![feature(array_chunks)]
#![feature(array_methods)]
#![feature(array_try_from_fn)]
#![feature(array_windows)]
#![feature(bigint_helper_methods)]
#![feature(cell_update)]
#![feature(const_align_of_val_raw)]
#![feature(const_align_offset)]
#![feature(const_array_from_ref)]
#![feature(const_assume)]
#![feature(const_black_box)]
#![feature(const_caller_location)]
#![feature(const_cell_into_inner)]
#![feature(const_hash)]
#![feature(const_heap)]
#![feature(const_ip)]
#![feature(const_ipv4)]
#![feature(const_ipv6)]
#![feature(const_likely)]
#![feature(const_location_fields)]
#![feature(const_maybe_uninit_as_mut_ptr)]
#![feature(const_maybe_uninit_assume_init_read)]
#![feature(const_mut_refs)]
#![feature(const_nonnull_new)]
#![feature(const_option)]
#![feature(const_option_ext)]
#![feature(const_pin)]
#![feature(const_pointer_byte_offsets)]
#![feature(const_pointer_is_aligned)]
#![feature(const_ptr_as_ref)]
#![feature(const_ptr_write)]
#![feature(const_result)]
#![feature(const_slice_from_ref)]
#![feature(const_trait_impl)]
#![feature(const_waker)]
#![feature(core_intrinsics)]
#![feature(core_private_bignum)]
#![feature(core_private_diy_float)]
#![feature(dec2flt)]
#![feature(div_duration)]
#![feature(duration_constants)]
#![feature(duration_consts_float)]
#![feature(exact_size_is_empty)]
#![feature(extern_types)]
#![feature(float_minimum_maximum)]
#![feature(flt2dec)]
#![feature(fmt_internals)]
#![feature(future_join)]
#![feature(generic_assert_internals)]
#![feature(get_many_mut)]
#![feature(hasher_prefixfree_extras)]
#![feature(hashmap_internals)]
#![feature(inline_const)]
#![feature(int_roundings)]
#![feature(ip)]
#![feature(ip_in_core)]
#![feature(is_ascii_octdigit)]
#![feature(is_sorted)]
#![feature(iter_advance_by)]
#![feature(iter_array_chunks)]
#![feature(iter_collect_into)]
#![feature(iter_intersperse)]
#![feature(iter_is_partitioned)]
#![feature(iter_next_chunk)]
#![feature(iter_order_by)]
#![feature(iter_partition_in_place)]
#![feature(iter_repeat_n)]
#![feature(iterator_try_collect)]
#![feature(iterator_try_reduce)]
#![feature(layout_for_ptr)]
#![feature(lazy_cell)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_uninit_array_transpose)]
#![feature(maybe_uninit_write_slice)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(num_midpoint)]
#![feature(numfmt)]
#![feature(offset_of)]
#![feature(pattern)]
#![feature(pointer_byte_offsets)]
#![feature(pointer_is_aligned)]
#![feature(portable_simd)]
#![feature(provide_any)]
#![feature(ptr_metadata)]
#![feature(slice_flatten)]
#![feature(slice_from_ptr_range)]
#![feature(slice_group_by)]
#![feature(slice_internals)]
#![feature(slice_partition_dedup)]
#![feature(slice_take)]
#![feature(sort_internals)]
#![feature(split_array)]
#![feature(split_as_slice)]
#![feature(std_internals)]
#![feature(step_trait)]
#![feature(str_internals)]
#![feature(strict_provenance)]
#![feature(strict_provenance_atomic_ptr)]
#![feature(test)]
#![feature(trusted_len)]
#![feature(trusted_random_access)]
#![feature(try_blocks)]
#![feature(try_find)]
#![feature(try_trait_v2)]
#![feature(unsize)]
#![feature(unsized_tuple_coercion)]
#![feature(unwrap_infallible)]
#![feature(utf8_chunks)]
#![feature(waker_getters)]
// tidy-alphabetical-end
#![cfg_attr(target_has_atomic = "128", feature(integer_atomics))]

extern crate test;

mod alloc;
mod any;
mod array;
mod ascii;
mod asserting;
mod atomic;
mod bool;
mod cell;
mod char;
mod clone;
mod cmp;
mod const_ptr;
mod convert;
mod fmt;
mod future;
mod hash;
mod intrinsics;
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
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    core::panic::Location::caller().hash(&mut hasher);
    let hc64 = hasher.finish();
    let seed_vec = hc64.to_le_bytes().into_iter().chain(0u8..8).collect::<Vec<u8>>();
    let seed: [u8; 16] = seed_vec.as_slice().try_into().unwrap();
    rand::SeedableRng::from_seed(seed)
}

#![feature(alloc_layout_extra)]
#![feature(array_chunks)]
#![feature(array_methods)]
#![feature(array_windows)]
#![feature(bigint_helper_methods)]
#![feature(cell_update)]
#![feature(const_align_offset)]
#![feature(const_assume)]
#![feature(const_align_of_val_raw)]
#![feature(const_black_box)]
#![feature(const_bool_to_option)]
#![feature(const_caller_location)]
#![feature(const_cell_into_inner)]
#![feature(const_convert)]
#![feature(const_hash)]
#![feature(const_heap)]
#![feature(const_maybe_uninit_as_mut_ptr)]
#![feature(const_maybe_uninit_assume_init_read)]
#![feature(const_nonnull_new)]
#![feature(const_num_from_num)]
#![feature(const_pointer_byte_offsets)]
#![feature(const_pointer_is_aligned)]
#![feature(const_ptr_as_ref)]
#![feature(const_ptr_read)]
#![feature(const_ptr_write)]
#![feature(const_trait_impl)]
#![feature(const_likely)]
#![feature(const_location_fields)]
#![feature(core_intrinsics)]
#![feature(core_private_bignum)]
#![feature(core_private_diy_float)]
#![feature(dec2flt)]
#![feature(div_duration)]
#![feature(duration_consts_float)]
#![feature(duration_constants)]
#![feature(exact_size_is_empty)]
#![feature(extern_types)]
#![feature(flt2dec)]
#![feature(fmt_internals)]
#![feature(float_minimum_maximum)]
#![feature(future_join)]
#![feature(generic_assert_internals)]
#![feature(array_try_from_fn)]
#![feature(hasher_prefixfree_extras)]
#![feature(hashmap_internals)]
#![feature(try_find)]
#![feature(inline_const)]
#![feature(is_sorted)]
#![feature(layout_for_ptr)]
#![feature(pattern)]
#![feature(sort_internals)]
#![feature(slice_take)]
#![feature(slice_from_ptr_range)]
#![feature(split_as_slice)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_write_slice)]
#![feature(maybe_uninit_uninit_array_transpose)]
#![feature(min_specialization)]
#![feature(numfmt)]
#![feature(step_trait)]
#![feature(str_internals)]
#![feature(std_internals)]
#![feature(test)]
#![feature(trusted_len)]
#![feature(try_blocks)]
#![feature(try_trait_v2)]
#![feature(slice_internals)]
#![feature(slice_partition_dedup)]
#![feature(ip)]
#![feature(ip_in_core)]
#![feature(iter_advance_by)]
#![feature(iter_array_chunks)]
#![feature(iter_collect_into)]
#![feature(iter_partition_in_place)]
#![feature(iter_intersperse)]
#![feature(iter_is_partitioned)]
#![feature(iter_next_chunk)]
#![feature(iter_order_by)]
#![feature(iter_repeat_n)]
#![feature(iterator_try_collect)]
#![feature(iterator_try_reduce)]
#![feature(const_ip)]
#![feature(const_ipv4)]
#![feature(const_ipv6)]
#![feature(const_mut_refs)]
#![feature(const_pin)]
#![feature(const_waker)]
#![feature(never_type)]
#![feature(unwrap_infallible)]
#![feature(pointer_byte_offsets)]
#![feature(pointer_is_aligned)]
#![feature(portable_simd)]
#![feature(ptr_metadata)]
#![feature(once_cell)]
#![feature(option_result_contains)]
#![feature(unsized_tuple_coercion)]
#![feature(const_option)]
#![feature(const_option_ext)]
#![feature(const_result)]
#![feature(integer_atomics)]
#![feature(int_roundings)]
#![feature(slice_group_by)]
#![feature(split_array)]
#![feature(strict_provenance)]
#![feature(strict_provenance_atomic_ptr)]
#![feature(trusted_random_access)]
#![feature(unsize)]
#![feature(const_array_from_ref)]
#![feature(const_slice_from_ref)]
#![feature(waker_getters)]
#![feature(slice_flatten)]
#![feature(provide_any)]
#![feature(utf8_chunks)]
#![feature(is_ascii_octdigit)]
#![feature(get_many_mut)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(fuzzy_provenance_casts)]

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

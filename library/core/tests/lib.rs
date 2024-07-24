#![feature(alloc_layout_extra)]
#![feature(array_chunks)]
#![feature(array_ptr_get)]
#![feature(array_windows)]
#![feature(ascii_char)]
#![feature(ascii_char_variants)]
#![feature(async_iter_from_iter)]
#![feature(async_iterator)]
#![feature(bigint_helper_methods)]
#![feature(cell_update)]
#![feature(clone_to_uninit)]
#![feature(const_align_offset)]
#![feature(const_align_of_val_raw)]
#![feature(const_black_box)]
#![feature(const_cell_into_inner)]
#![feature(const_hash)]
#![feature(const_heap)]
#![feature(const_intrinsic_copy)]
#![feature(const_int_from_str)]
#![feature(const_maybe_uninit_as_mut_ptr)]
#![feature(const_nonnull_new)]
#![feature(const_pointer_is_aligned)]
#![feature(const_ptr_as_ref)]
#![feature(const_ptr_write)]
#![feature(const_three_way_compare)]
#![feature(const_trait_impl)]
#![feature(const_likely)]
#![feature(core_intrinsics)]
#![feature(core_io_borrowed_buf)]
#![feature(core_private_bignum)]
#![feature(core_private_diy_float)]
#![feature(dec2flt)]
#![feature(duration_consts_float)]
#![feature(duration_constants)]
#![feature(duration_constructors)]
#![feature(exact_size_is_empty)]
#![feature(extern_types)]
#![feature(freeze)]
#![feature(flt2dec)]
#![feature(fmt_internals)]
#![feature(float_minimum_maximum)]
#![feature(future_join)]
#![feature(generic_assert_internals)]
#![feature(array_try_from_fn)]
#![feature(hasher_prefixfree_extras)]
#![feature(hashmap_internals)]
#![feature(try_find)]
#![feature(is_sorted)]
#![feature(layout_for_ptr)]
#![feature(pattern)]
#![feature(slice_take)]
#![feature(slice_from_ptr_range)]
#![feature(slice_split_once)]
#![feature(split_as_slice)]
#![feature(maybe_uninit_fill)]
#![feature(maybe_uninit_write_slice)]
#![feature(maybe_uninit_uninit_array_transpose)]
#![feature(min_specialization)]
#![feature(noop_waker)]
#![feature(numfmt)]
#![feature(num_midpoint)]
#![feature(offset_of_nested)]
#![feature(isqrt)]
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
#![feature(iter_advance_by)]
#![feature(iter_array_chunks)]
#![feature(iter_chain)]
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
#![feature(pointer_is_aligned_to)]
#![feature(portable_simd)]
#![feature(ptr_metadata)]
#![feature(unsized_tuple_coercion)]
#![feature(const_option)]
#![feature(const_option_ext)]
#![feature(const_result)]
#![cfg_attr(target_has_atomic = "128", feature(integer_atomics))]
#![cfg_attr(test, feature(cfg_match))]
#![feature(int_roundings)]
#![feature(split_array)]
#![feature(strict_provenance)]
#![feature(strict_provenance_atomic_ptr)]
#![feature(trusted_random_access)]
#![feature(unsize)]
#![feature(const_array_from_ref)]
#![feature(const_slice_from_ref)]
#![feature(waker_getters)]
#![feature(error_generic_member_access)]
#![feature(trait_upcasting)]
#![feature(is_ascii_octdigit)]
#![feature(get_many_mut)]
#![feature(iter_map_windows)]
#![allow(internal_features)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(fuzzy_provenance_casts)]

mod alloc;
mod any;
mod array;
mod ascii;
mod asserting;
mod async_iter;
mod atomic;
mod bool;
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
mod intrinsics;
mod io;
mod iter;
mod lazy;
#[cfg(test)]
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

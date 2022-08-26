#![feature(alloc_layout_extra)]
#![feature(array_chunks)]
#![feature(array_methods)]
#![feature(array_windows)]
#![feature(bench_black_box)]
#![feature(cell_update)]
#![feature(const_assume)]
#![feature(const_black_box)]
#![feature(const_bool_to_option)]
#![feature(const_cell_into_inner)]
#![feature(const_convert)]
#![feature(const_heap)]
#![feature(const_maybe_uninit_as_mut_ptr)]
#![feature(const_maybe_uninit_assume_init_read)]
#![feature(const_nonnull_new)]
#![feature(const_num_from_num)]
#![feature(const_pointer_byte_offsets)]
#![feature(const_ptr_as_ref)]
#![feature(const_ptr_read)]
#![feature(const_ptr_write)]
#![feature(const_trait_impl)]
#![feature(const_likely)]
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
#![feature(pattern)]
#![feature(pin_macro)]
#![feature(sort_internals)]
#![feature(slice_take)]
#![feature(slice_from_ptr_range)]
#![feature(split_as_slice)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_write_slice)]
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
#![feature(int_log)]
#![feature(iter_advance_by)]
#![feature(iter_array_chunks)]
#![feature(iter_collect_into)]
#![feature(iter_partition_in_place)]
#![feature(iter_intersperse)]
#![feature(iter_is_partitioned)]
#![feature(iter_next_chunk)]
#![feature(iter_order_by)]
#![feature(iterator_try_collect)]
#![feature(iterator_try_reduce)]
#![feature(const_mut_refs)]
#![feature(const_pin)]
#![feature(never_type)]
#![feature(unwrap_infallible)]
#![feature(pointer_byte_offsets)]
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
#![feature(unzip_option)]
#![feature(const_array_from_ref)]
#![feature(const_slice_from_ref)]
#![feature(waker_getters)]
#![feature(slice_flatten)]
#![feature(provide_any)]
#![feature(utf8_chunks)]
#![deny(unsafe_op_in_unsafe_fn)]

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
mod nonzero;
mod num;
mod ops;
mod option;
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

#![feature(alloc_layout_extra)]
#![feature(array_chunks)]
#![feature(array_methods)]
#![feature(array_windows)]
#![feature(bool_to_option)]
#![feature(box_syntax)]
#![feature(cell_update)]
#![feature(cfg_panic)]
#![feature(cfg_target_has_atomic)]
#![feature(const_assume)]
#![feature(const_cell_into_inner)]
#![feature(const_maybe_uninit_assume_init)]
#![cfg_attr(bootstrap, feature(const_panic))]
#![feature(const_ptr_read)]
#![feature(const_ptr_write)]
#![feature(const_ptr_offset)]
#![feature(const_trait_impl)]
#![feature(const_num_from_num)]
#![feature(core_intrinsics)]
#![feature(core_private_bignum)]
#![feature(core_private_diy_float)]
#![feature(dec2flt)]
#![feature(div_duration)]
#![feature(duration_consts_2)]
#![feature(duration_constants)]
#![feature(exact_size_is_empty)]
#![feature(extern_types)]
#![feature(flt2dec)]
#![feature(fmt_internals)]
#![feature(array_from_fn)]
#![feature(hashmap_internals)]
#![feature(try_find)]
#![feature(is_sorted)]
#![feature(pattern)]
#![feature(sort_internals)]
#![feature(slice_partition_at_index)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_extra)]
#![feature(maybe_uninit_write_slice)]
#![feature(min_specialization)]
#![feature(numfmt)]
#![feature(step_trait)]
#![feature(str_internals)]
#![feature(test)]
#![feature(trusted_len)]
#![feature(try_trait_v2)]
#![feature(slice_internals)]
#![feature(slice_partition_dedup)]
#![feature(int_log)]
#![feature(iter_advance_by)]
#![feature(iter_partition_in_place)]
#![feature(iter_intersperse)]
#![feature(iter_is_partitioned)]
#![feature(iter_order_by)]
#![feature(const_mut_refs)]
#![feature(const_pin)]
#![feature(const_slice_from_raw_parts)]
#![feature(const_raw_ptr_deref)]
#![feature(never_type)]
#![feature(unwrap_infallible)]
#![feature(option_result_unwrap_unchecked)]
#![feature(result_into_ok_or_err)]
#![feature(ptr_metadata)]
#![feature(once_cell)]
#![feature(unsized_tuple_coercion)]
#![feature(const_option)]
#![feature(const_result)]
#![feature(integer_atomics)]
#![feature(int_roundings)]
#![feature(slice_group_by)]
#![feature(trusted_random_access)]
#![feature(unsize)]
#![feature(unzip_option)]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate test;

mod alloc;
mod any;
mod array;
mod ascii;
mod atomic;
mod bool;
mod cell;
mod char;
mod clone;
mod cmp;
mod const_ptr;
mod fmt;
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
mod ptr;
mod result;
mod slice;
mod str;
mod str_lossy;
mod task;
mod time;
mod tuple;
mod unicode;

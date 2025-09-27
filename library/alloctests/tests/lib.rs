#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
#![feature(iter_array_chunks)]
#![feature(assert_matches)]
#![feature(wtf8_internals)]
#![feature(char_max_len)]
#![feature(cow_is_borrowed)]
#![feature(core_intrinsics)]
#![feature(downcast_unchecked)]
#![feature(exact_size_is_empty)]
#![feature(hashmap_internals)]
#![feature(int_format_into)]
#![feature(linked_list_cursors)]
#![feature(map_try_insert)]
#![feature(pattern)]
#![feature(trusted_len)]
#![feature(try_reserve_kind)]
#![feature(try_with_capacity)]
#![feature(unboxed_closures)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(binary_heap_drain_sorted)]
#![feature(slice_ptr_get)]
#![feature(inplace_iteration)]
#![feature(iter_advance_by)]
#![feature(iter_next_chunk)]
#![feature(slice_partition_dedup)]
#![feature(string_from_utf8_lossy_owned)]
#![feature(string_remove_matches)]
#![feature(const_btree_len)]
#![feature(const_trait_impl)]
#![feature(panic_update_hook)]
#![feature(pointer_is_aligned_to)]
#![feature(test)]
#![feature(thin_box)]
#![feature(drain_keep_rest)]
#![feature(local_waker)]
#![feature(str_as_str)]
#![feature(strict_provenance_lints)]
#![feature(vec_deque_pop_if)]
#![feature(vec_deque_truncate_front)]
#![feature(unique_rc_arc)]
#![feature(macro_metavar_expr_concat)]
#![feature(vec_peek_mut)]
#![feature(vec_try_remove)]
#![allow(internal_features)]
#![deny(fuzzy_provenance_casts)]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;

use std::hash::{DefaultHasher, Hash, Hasher};

mod alloc_test;
mod arc;
mod autotraits;
mod borrow;
mod boxed;
mod btree_set_hash;
mod c_str;
mod c_str2;
mod collections;
mod const_fns;
mod cow_str;
mod fmt;
mod heap;
mod linked_list;
mod misc_tests;
mod num;
mod rc;
mod slice;
mod sort;
mod str;
mod string;
mod sync;
mod task;
mod testing;
mod thin_box;
mod vec;
mod vec_deque;

fn hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

/// Copied from `std::test_helpers::test_rng`, since these tests rely on the
/// seed not being the same for every RNG invocation too.
fn test_rng() -> rand_xorshift::XorShiftRng {
    use std::hash::{BuildHasher, Hash, Hasher};
    let mut hasher = std::hash::RandomState::new().build_hasher();
    std::panic::Location::caller().hash(&mut hasher);
    let hc64 = hasher.finish();
    let seed_vec = hc64.to_le_bytes().into_iter().chain(0u8..8).collect::<Vec<u8>>();
    let seed: [u8; 16] = seed_vec.as_slice().try_into().unwrap();
    rand::SeedableRng::from_seed(seed)
}

#[test]
fn test_boxed_hasher() {
    let ordinary_hash = hash(&5u32);

    let mut hasher_1 = Box::new(DefaultHasher::new());
    5u32.hash(&mut hasher_1);
    assert_eq!(ordinary_hash, hasher_1.finish());

    let mut hasher_2 = Box::new(DefaultHasher::new()) as Box<dyn Hasher>;
    5u32.hash(&mut hasher_2);
    assert_eq!(ordinary_hash, hasher_2.finish());
}

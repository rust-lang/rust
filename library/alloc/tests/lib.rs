#![feature(allocator_api)]
#![feature(assert_matches)]
#![feature(box_syntax)]
#![feature(cow_is_borrowed)]
#![feature(const_cow_is_borrowed)]
#![feature(drain_filter)]
#![feature(exact_size_is_empty)]
#![feature(new_uninit)]
#![feature(pattern)]
#![feature(trusted_len)]
#![feature(try_reserve)]
#![feature(try_reserve_kind)]
#![feature(unboxed_closures)]
#![feature(associated_type_bounds)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(binary_heap_drain_sorted)]
#![feature(slice_ptr_get)]
#![feature(binary_heap_retain)]
#![feature(binary_heap_as_slice)]
#![feature(inplace_iteration)]
#![feature(iter_map_while)]
#![feature(slice_group_by)]
#![feature(slice_partition_dedup)]
#![feature(vec_spare_capacity)]
#![feature(string_remove_matches)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod arc;
mod binary_heap;
mod borrow;
mod boxed;
mod btree_set_hash;
mod cow_str;
mod fmt;
mod heap;
mod linked_list;
mod rc;
mod slice;
mod str;
mod string;
mod vec;
mod vec_deque;

fn hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

// FIXME: Instantiated functions with i128 in the signature is not supported in Emscripten.
// See https://github.com/kripken/emscripten-fastcomp/issues/169
#[cfg(not(target_os = "emscripten"))]
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

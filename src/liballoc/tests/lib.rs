#![feature(allocator_api)]
#![feature(box_syntax)]
#![feature(btree_drain_filter)]
#![feature(drain_filter)]
#![feature(exact_size_is_empty)]
#![feature(map_first_last)]
#![feature(new_uninit)]
#![feature(pattern)]
#![feature(trusted_len)]
#![feature(try_reserve)]
#![feature(unboxed_closures)]
#![feature(associated_type_bounds)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(binary_heap_drain_sorted)]
#![feature(vec_remove_item)]
#![feature(split_inclusive)]
#![feature(binary_heap_retain)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod arc;
mod binary_heap;
mod borrow;
mod boxed;
mod btree;
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

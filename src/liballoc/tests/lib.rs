#![feature(allocator_api)]
#![feature(box_syntax)]
#![feature(drain_filter)]
#![feature(exact_size_is_empty)]
#![feature(pattern)]
#![feature(repeat_generic_slice)]
#![feature(try_reserve)]
#![feature(unboxed_closures)]
#![deny(rust_2018_idioms)]

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

mod arc;
mod binary_heap;
mod btree;
mod cow_str;
mod fmt;
mod heap;
mod linked_list;
mod rc;
mod slice;
mod str;
mod string;
mod vec_deque;
mod vec;

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

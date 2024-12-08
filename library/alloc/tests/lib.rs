#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
#![feature(iter_array_chunks)]
#![feature(assert_matches)]
#![feature(btree_extract_if)]
#![feature(cow_is_borrowed)]
#![feature(core_intrinsics)]
#![feature(extract_if)]
#![feature(exact_size_is_empty)]
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
#![feature(round_char_boundary)]
#![feature(slice_partition_dedup)]
#![feature(string_from_utf8_lossy_owned)]
#![feature(string_remove_matches)]
#![feature(const_btree_len)]
#![feature(const_trait_impl)]
#![feature(const_str_from_utf8)]
#![feature(panic_update_hook)]
#![feature(pointer_is_aligned_to)]
#![feature(thin_box)]
#![feature(drain_keep_rest)]
#![feature(local_waker)]
#![feature(strict_provenance_lints)]
#![feature(vec_pop_if)]
#![feature(unique_rc_arc)]
#![feature(macro_metavar_expr_concat)]
#![allow(internal_features)]
#![deny(fuzzy_provenance_casts)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::hash::{DefaultHasher, Hash, Hasher};

mod arc;
mod autotraits;
mod borrow;
mod boxed;
mod btree_set_hash;
mod c_str;
mod const_fns;
mod cow_str;
mod fmt;
mod heap;
mod linked_list;
mod rc;
mod slice;
mod sort;
mod str;
mod string;
mod task;
mod thin_box;
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

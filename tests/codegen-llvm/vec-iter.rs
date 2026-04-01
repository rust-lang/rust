//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(exact_size_is_empty)]

use std::vec;

// CHECK-LABEL: @vec_iter_len_nonnull
#[no_mangle]
pub fn vec_iter_len_nonnull(it: &vec::IntoIter<u8>) -> usize {
    // CHECK: load ptr
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: load ptr
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: sub nuw
    // CHECK: ret
    it.len()
}

// CHECK-LABEL: @vec_iter_is_empty_nonnull
#[no_mangle]
pub fn vec_iter_is_empty_nonnull(it: &vec::IntoIter<u8>) -> bool {
    // CHECK: load ptr
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: load ptr
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: ret
    it.is_empty()
}

// CHECK-LABEL: @vec_iter_next_nonnull
#[no_mangle]
pub fn vec_iter_next_nonnull(it: &mut vec::IntoIter<u8>) -> Option<u8> {
    // CHECK: load ptr
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: load ptr
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: ret
    it.next()
}

// CHECK-LABEL: @vec_iter_next_back_nonnull
#[no_mangle]
pub fn vec_iter_next_back_nonnull(it: &mut vec::IntoIter<u8>) -> Option<u8> {
    // CHECK: load ptr
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: load ptr
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK: ret
    it.next_back()
}

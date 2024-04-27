//@ compile-flags: -O
//@ min-llvm-version: 18 (which added `dead_on_unwind`)
#![crate_type = "lib"]
#![feature(exact_size_is_empty)]
#![feature(iter_advance_by)]
#![feature(iter_next_chunk)]

use std::{array, vec};

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

// CHECK-LABEL: @vec_iter_next_transfers_ownership
#[no_mangle]
pub fn vec_iter_next_transfers_ownership(it: &mut vec::IntoIter<Box<i32>>) -> Option<Box<i32>> {
    // CHECK-NOT: __rust_dealloc
    it.next()
}

// CHECK-LABEL: @vec_iter_advance_drops_item
#[no_mangle]
pub fn vec_iter_advance_drops_item(it: &mut vec::IntoIter<Box<i32>>) {
    // CHECK-NOT: __rust_dealloc
    // CHECK: call void @__rust_dealloc
    // CHECK-SAME: noundef 4
    // CHECK-SAME: noundef 4
    // CHECK-NOT: __rust_dealloc
    _ = it.advance_by(1);
}

// CHECK-LABEL: @vec_iter_next_chunk_short
// CHECK-SAME: ptr{{.+}}%[[RET:.+]],
#[no_mangle]
pub fn vec_iter_next_chunk_short(
    it: &mut vec::IntoIter<u8>,
) -> Result<[u8; 4], array::IntoIter<u8, 4>> {
    // CHECK-NOT: alloca
    // CHECK: %[[ACTUAL_LEN:.+]] = sub nuw

    // CHECK: %[[OUT1:.+]] = getelementptr inbounds i8, ptr %[[RET]]
    // CHECK: call void @llvm.memcpy{{.+}}(ptr{{.+}}%[[OUT1]],{{.+}} %[[ACTUAL_LEN]], i1 false)
    // CHECK: br

    // CHECK: %[[FULL:.+]] = load i32,
    // CHECK: %[[OUT2:.+]] = getelementptr inbounds i8, ptr %[[RET]]
    // CHECK: store i32 %[[FULL]], ptr %[[OUT2]]
    // CHECK: br
    it.next_chunk::<4>()
}

// CHECK-LABEL: @vec_iter_next_chunk_long
// CHECK-SAME: ptr{{.+}}%[[RET:.+]],
#[no_mangle]
pub fn vec_iter_next_chunk_long(
    it: &mut vec::IntoIter<u8>,
) -> Result<[u8; 123], array::IntoIter<u8, 123>> {
    // CHECK-NOT: alloca
    // CHECK: %[[ACTUAL_LEN:.+]] = sub nuw

    // CHECK: %[[OUT1:.+]] = getelementptr inbounds i8, ptr %[[RET]]
    // CHECK: call void @llvm.memcpy{{.+}}(ptr{{.+}}%[[OUT1]],{{.+}} %[[ACTUAL_LEN]], i1 false)
    // CHECK: br

    // CHECK: %[[OUT2:.+]] = getelementptr inbounds i8, ptr %[[RET]]
    // CHECK: call void @llvm.memcpy{{.+}}(ptr{{.+}}%[[OUT2]],{{.+}} 123, i1 false)
    // CHECK: br
    it.next_chunk::<123>()
}

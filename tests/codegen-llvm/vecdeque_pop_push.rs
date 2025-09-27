//@ compile-flags: -Copt-level=3
//@ revisions: new old
//@ [old] max-llvm-major-version: 21
//@ [new] min-llvm-version: 22

#![crate_type = "lib"]

use std::collections::VecDeque;

#[no_mangle]
// CHECK-LABEL: @noop_back(
pub fn noop_back(v: &mut VecDeque<u8>) {
    // CHECK-NOT: grow
    // old: tail call void @llvm.assume
    // CHECK-NOT: grow
    // CHECK: ret
    if let Some(x) = v.pop_back() {
        v.push_back(x);
    }
}

#[no_mangle]
// CHECK-LABEL: @noop_front(
pub fn noop_front(v: &mut VecDeque<u8>) {
    // CHECK-NOT: grow
    // CHECK: tail call void @llvm.assume
    // CHECK-NOT: grow
    // CHECK: ret
    if let Some(x) = v.pop_front() {
        v.push_front(x);
    }
}

#[no_mangle]
// CHECK-LABEL: @move_byte_front_to_back(
pub fn move_byte_front_to_back(v: &mut VecDeque<u8>) {
    // CHECK-NOT: grow
    // CHECK: tail call void @llvm.assume
    // CHECK-NOT: grow
    // CHECK: ret
    if let Some(x) = v.pop_front() {
        v.push_back(x);
    }
}

#[no_mangle]
// CHECK-LABEL: @move_byte_back_to_front(
pub fn move_byte_back_to_front(v: &mut VecDeque<u8>) {
    // CHECK-NOT: grow
    // CHECK: tail call void @llvm.assume
    // CHECK-NOT: grow
    // CHECK: ret
    if let Some(x) = v.pop_back() {
        v.push_front(x);
    }
}

#[no_mangle]
// CHECK-LABEL: @push_back_byte(
pub fn push_back_byte(v: &mut VecDeque<u8>) {
    // CHECK: call {{.*}}grow
    v.push_back(3);
}

#[no_mangle]
// CHECK-LABEL: @push_front_byte(
pub fn push_front_byte(v: &mut VecDeque<u8>) {
    // CHECK: call {{.*}}grow
    v.push_front(3);
}

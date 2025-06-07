//@ compile-flags: -Z merge-functions=disabled

#![crate_type = "lib"]

use std::{rc, sync};

#[no_mangle]
pub fn array_of_rc_weak() -> [rc::Weak<u32>; 100] {
    // CHECK-LABEL: @array_of_rc_weak(
    // CHECK-NEXT: start:
    // CHECK-NEXT: call void @llvm.memset.
    // CHECK-NEXT: ret void
    [(); 100].map(|()| rc::Weak::new())
}

#[no_mangle]
pub fn array_of_sync_weak() -> [sync::Weak<u32>; 100] {
    // CHECK-LABEL: @array_of_sync_weak(
    // CHECK-NEXT: start:
    // CHECK-NEXT: call void @llvm.memset.
    // CHECK-NEXT: ret void
    [(); 100].map(|()| sync::Weak::new())
}

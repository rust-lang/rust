//@ compile-flags: -O -Z merge-functions=disabled

#![crate_type = "lib"]

use std::rc::{self, Rc};
use std::sync::{self, Arc};

// Ensures that we can create an array of `Weak`s using `memset`.

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

// Asserts that converting from `&Option<Rc<T>>` and `&Option<Arc<T>>` to `Option<&T>` required
// checking for `None`.

#[no_mangle]
pub fn option_rc_as_deref_using_cmp(rc: &Option<Rc<u32>>) -> Option<&u32> {
    // CHECK-LABEL: @option_rc_as_deref_using_cmp(ptr
    // CHECK-NEXT: start:
    // CHECK-NEXT: load ptr, ptr
    // CHECK-NEXT: icmp
    // CHECK: ret ptr
    rc.as_deref()
}

#[no_mangle]
pub fn option_arc_as_deref_using_cmp(arc: &Option<Arc<u32>>) -> Option<&u32> {
    // CHECK-LABEL: @option_arc_as_deref_using_cmp(ptr
    // CHECK-NEXT: start:
    // CHECK-NEXT: load ptr, ptr
    // CHECK-NEXT: icmp
    // CHECK: ret ptr
    arc.as_deref()
}

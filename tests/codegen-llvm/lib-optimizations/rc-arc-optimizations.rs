//@ compile-flags: -O -Z merge-functions=disabled

#![crate_type = "lib"]

use std::rc::{self, Rc};
use std::sync::{self, Arc};

// Ensures that we can create array of `Weak`s using `memset`.

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

// Ensures that we convert `&Option<Rc<T>>` and `&Option<Arc<T>>` to `Option<&T>` without checking
// for `None`.

#[no_mangle]
pub fn option_rc_as_deref_no_cmp(rc: &Option<Rc<u32>>) -> Option<&u32> {
    // CHECK-LABEL: @option_rc_as_deref_no_cmp(ptr
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[RC:.+]] = load ptr, ptr %rc
    // CHECK-NEXT: ret ptr %[[RC]]
    rc.as_deref()
}

#[no_mangle]
pub fn option_arc_as_deref_no_cmp(arc: &Option<Arc<u32>>) -> Option<&u32> {
    // CHECK-LABEL: @option_arc_as_deref_no_cmp(ptr
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[ARC:.+]] = load ptr, ptr %arc
    // CHECK-NEXT: ret ptr %[[ARC]]
    arc.as_deref()
}

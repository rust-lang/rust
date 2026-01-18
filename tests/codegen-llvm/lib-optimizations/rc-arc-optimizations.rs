//@ compile-flags: -O -Z merge-functions=disabled

#![crate_type = "lib"]

use std::rc::{self, Rc};

// Ensures that we can create array of `Weak`s using `memset`.

#[no_mangle]
pub fn array_of_rc_weak() -> [rc::Weak<u32>; 100] {
    // CHECK-LABEL: @array_of_rc_weak(
    // CHECK-NEXT: start:
    // CHECK-NEXT: call void @llvm.memset.
    // CHECK-NEXT: ret void
    [(); 100].map(|()| rc::Weak::new())
}

// Ensures that we convert `&Option<Rc<T>>` to `Option<&T>` without checking for `None`.

#[no_mangle]
pub fn option_rc_as_deref_no_cmp(rc: &Option<Rc<u32>>) -> Option<&u32> {
    // CHECK-LABEL: @option_rc_as_deref_no_cmp(ptr
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[RC:.+]] = load ptr, ptr %rc
    // CHECK-NEXT: ret ptr %[[RC]]
    rc.as_deref()
}

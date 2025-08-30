// Ensures that we can acquire `Option<&T>` from `&Option<Rc<T>>` without checking for `None`.

//@ compile-flags: -Z merge-functions=disabled

#![crate_type = "lib"]

use std::rc::Rc;

#[no_mangle]
pub fn option_rc_as_deref_no_cmp(rc: &Option<Rc<u32>>) -> Option<&u32> {
    // CHECK-LABEL: @option_rc_as_deref_no_cmp(ptr
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[RC:.+]] = load ptr, ptr %rc
    // CHECK-NEXT: ret ptr %[[RC]]
    rc.as_deref()
}

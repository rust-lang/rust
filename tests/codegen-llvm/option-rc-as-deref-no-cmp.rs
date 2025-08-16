//@ compile-flags: -Z merge-functions=disabled

#![crate_type = "lib"]

use std::rc::Rc;
use std::sync::Arc;

#[no_mangle]
pub fn option_arc_as_deref_no_cmp(arc: &Option<Arc<u32>>) -> Option<&u32> {
    // CHECK-LABEL: @option_arc_as_deref_no_cmp(ptr
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[ARC:.+]] = load ptr, ptr %arc
    // CHECK-NEXT: ret ptr %[[ARC]]
    arc.as_deref()
}

#[no_mangle]
pub fn option_rc_as_deref_no_cmp(rc: &Option<Rc<u32>>) -> Option<&u32> {
    // CHECK-LABEL: @option_rc_as_deref_no_cmp(ptr
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[RC:.+]] = load ptr, ptr %rc
    // CHECK-NEXT: ret ptr %[[RC]]
    rc.as_deref()
}

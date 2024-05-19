//@ no-prefer-dynamic
//@ compile-flags: -C debuginfo=2
#![crate_type="rlib"]

extern crate b_reexport_obj;
use b_reexport_obj::Object;

pub fn another_dyn_debug() {
    let ref u = 1_u32;
    let _d = &u as &dyn crate::Object;
    _d.method()
}

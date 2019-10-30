// compile-flags: -C debuginfo=2 --crate-type=rlib

extern crate reexport_obj;
use reexport_obj::Object;

pub fn another_dyn_debug() {
    let ref u = 1_u32;
    let _d = &u as &dyn crate::Object;
    _d.method()
}

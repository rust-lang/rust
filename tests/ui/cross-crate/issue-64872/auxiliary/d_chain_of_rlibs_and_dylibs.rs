//@ compile-flags: -C debuginfo=2 -C prefer-dynamic

#![crate_type="rlib"]

extern crate c_another_vtable_for_obj;

pub fn chain() {
    c_another_vtable_for_obj::another_dyn_debug();
}

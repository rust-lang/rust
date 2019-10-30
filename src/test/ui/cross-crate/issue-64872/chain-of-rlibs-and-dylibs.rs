// note that these aux-build directives must be in this order: the later crates
// depend on the earlier ones.

// aux-build:def_obj.rs
// aux-build:rexport_obj.rs
// aux-build:another_vtable_for_obj.rs

extern crate another_vtable_for_obj;

pub fn main() {
    another_vtable_for_obj::another_dyn_debug();
}

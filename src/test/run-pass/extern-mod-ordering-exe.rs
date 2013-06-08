// aux-build:extern_mod_ordering_lib.rs
// xfail-fast

extern mod extern_mod_ordering_lib;

use extern_mod_ordering_lib::extern_mod_ordering_lib;

pub fn main() {
    extern_mod_ordering_lib::f();
}

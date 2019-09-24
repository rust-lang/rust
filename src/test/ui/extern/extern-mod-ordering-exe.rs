// run-pass
// aux-build:extern_mod_ordering_lib.rs

// pretty-expanded FIXME #23616

extern crate extern_mod_ordering_lib;

use extern_mod_ordering_lib::extern_mod_ordering_lib as the_lib;

pub fn main() {
    the_lib::f();
}

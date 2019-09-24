// run-pass
// aux-build:moves_based_on_type_lib.rs

// pretty-expanded FIXME #23616

extern crate moves_based_on_type_lib;
use moves_based_on_type_lib::f;

pub fn main() {
    f();
}

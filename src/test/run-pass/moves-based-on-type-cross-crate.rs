// xfail-fast
// aux-build:moves_based_on_type_lib.rs

extern mod moves_based_on_type_lib;
use moves_based_on_type_lib::f;

fn main() {
    f();
}

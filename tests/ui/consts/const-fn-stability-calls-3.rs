// Test use of const fn from another crate without a feature gate.

// check-pass
// aux-build:const_fn_lib.rs

extern crate const_fn_lib;

use const_fn_lib::foo;

fn main() {
    let x = foo(); // use outside a constant is ok
}

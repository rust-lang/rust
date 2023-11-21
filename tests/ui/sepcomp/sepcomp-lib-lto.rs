// run-pass
// Check that we can use `-C lto` when linking against libraries that were
// separately compiled.

// aux-build:sepcomp_lib.rs
// compile-flags: -C lto -g
// no-prefer-dynamic

extern crate sepcomp_lib;
use sepcomp_lib::a::one;
use sepcomp_lib::b::two;
use sepcomp_lib::c::three;

fn main() {
    assert_eq!(one(), 1);
    assert_eq!(two(), 2);
    assert_eq!(three(), 3);
}

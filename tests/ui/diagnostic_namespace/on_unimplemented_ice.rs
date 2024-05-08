//@ edition:2021
//@ compile-flags:--test
//@ aux-build:bad_on_unimplemented.rs

// Do not ICE when encountering a malformed `#[diagnostic::on_unimplemented]` annotation in a
// dependency when incorrectly used (#124651).

extern crate bad_on_unimplemented;

use bad_on_unimplemented::Test;

fn breakage<T: Test>(_: T) {}

#[test]
fn test() {
    breakage(1); //~ ERROR E0277
}

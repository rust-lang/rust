// Checks that the following does not ICE because `decorate` is incorrectly skipped.

//@ compile-flags: -Dunused_must_use -Awarnings --crate-type=lib

#[must_use]
fn f() {}

pub fn g() {
    f();
    //~^ ERROR unused return value
}

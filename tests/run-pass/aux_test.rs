// aux-build:dep.rs

// ignore-cross-compile
// TODO: The above accidentally also ignores this test against rustc even when are are not cross-compiling.

extern crate dep;

fn main() {
    dep::foo();
}

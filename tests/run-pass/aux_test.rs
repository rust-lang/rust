// aux-build:dep.rs
// This ignores the test against rustc, but runs it against miri:
// ignore-cross-compile

extern crate dep;

fn main() {
    dep::foo();
}

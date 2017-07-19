// aux-build:dep.rs

// ignore-cross-compile

extern crate dep;

fn main() {
    dep::foo();
}

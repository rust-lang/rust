// compile-pass
// aux-build:two_macros.rs

extern crate two_macros as core;

mod m {
    fn check() {
        core::m!(); // OK
    }
}

fn main() {}

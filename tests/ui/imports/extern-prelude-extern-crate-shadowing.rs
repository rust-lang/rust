// build-pass (FIXME(62277): could be check-pass?)
// aux-build:two_macros.rs

extern crate two_macros as core;

mod m {
    fn check() {
        core::m!(); // OK
    }
}

fn main() {}

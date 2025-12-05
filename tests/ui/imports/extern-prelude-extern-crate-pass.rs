//@ build-pass (FIXME(62277): could be check-pass?)
//@ aux-build:two_macros.rs

extern crate two_macros;

mod m {
    fn check() {
        two_macros::m!(); // OK
    }
}

fn main() {}

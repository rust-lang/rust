//@ no-prefer-dynamic
//@ compile-flags: -Ccodegen-units=2 --crate-type=lib

extern crate cgu_test;

pub mod a {
    pub fn a() {
        ::cgu_test::id(0);
    }
}
pub mod b {
    pub fn a() {
        ::cgu_test::id(0);
    }
}

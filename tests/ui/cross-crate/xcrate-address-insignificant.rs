//@ run-pass
//@ aux-build:xcrate_address_insignificant.rs


extern crate xcrate_address_insignificant as foo;

pub fn main() {
    assert_eq!(foo::foo::<f64>(), foo::bar());
}

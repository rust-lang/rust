//@ run-pass
//@ aux-build:xcrate_generic_fn_nested_return.rs

extern crate xcrate_generic_fn_nested_return as test;

pub fn main() {
    assert!(test::decode::<()>().is_err());
}

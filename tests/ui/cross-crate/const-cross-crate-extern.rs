//@ run-pass
//@ aux-build:cci_const.rs

#![allow(non_upper_case_globals)]
#![allow(unpredictable_function_pointer_comparisons)]

extern crate cci_const;
use cci_const::bar;
static foo: extern "C" fn() = bar;

pub fn main() {
    assert!(foo == bar);
}

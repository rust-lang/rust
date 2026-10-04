//! Regression test for <https://github.com/rust-lang/rust/issues/102190>

//@ check-pass
//@ proc-macro: generate-inherent-impl-with-input-spans.rs

#![deny(dead_code)]

extern crate generate_inherent_impl_with_input_spans;

use generate_inherent_impl_with_input_spans::Test;

#[allow(dead_code)]
#[derive(Test)]
#[repr(i8)]
pub(crate) enum MyEnum {
    MyValue,
}

fn main() {}

#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

const A: usize = *&5;

#[miri_run]
fn foo() -> usize {
    A
}

fn main() {}

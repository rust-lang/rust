#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

struct A;

#[miri_run]
fn zst_ret() -> A {
    A
}

#[miri_run]
fn use_zst() -> A {
    let a = A;
    a
}

fn main() {}

#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn empty() {}

#[miri_run]
fn unit_var() {
    let x = ();
    x
}

fn main() {}

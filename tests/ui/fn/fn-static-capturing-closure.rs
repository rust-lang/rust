#![feature(fn_static)]

use std::ops::FnStatic;

fn assert_fn_static(_: impl FnStatic) {}

fn main() {
    let captured = 42;
    assert_fn_static(|| captured);
    //~^ ERROR E0277
}

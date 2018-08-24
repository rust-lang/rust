#![feature(rustc_attrs)]

use std::ops::*;

#[derive(Copy, Clone)]
struct R(RangeTo<usize>);

#[rustc_error]
fn main() {} //~ ERROR success


#![feature(rustc_attrs)]

use std::ops::*;

#[derive(Copy, Clone)]
struct R(RangeToInclusive<usize>);

#[rustc_error]
fn main() {} //~ ERROR success


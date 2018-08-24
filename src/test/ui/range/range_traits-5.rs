#![feature(rustc_attrs)]

use std::ops::*;

#[derive(Copy, Clone)]
struct R(RangeFull);

#[rustc_error]
fn main() {} //~ ERROR success


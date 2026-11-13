// reject other attributes on autodiff function. (#152889)
#![feature(macro_attr)]
#![feature(autodiff)]

macro_rules! forward_referenced_attr {
    attr() {$($tt:tt)*} => {$($tt)*}
}

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(Active)]
#[forward_referenced_attr]
fn foo() {}
//~^ ERROR only builtin and autodiff attributes are supported on autodiff functions

fn main(){}

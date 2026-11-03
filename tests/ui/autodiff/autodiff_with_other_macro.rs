// reject other attributes on autodiff function. (#152889)
#![feature(macro_attr)]
#![feature(autodiff)]

macro_rules! forward_referenced_attr {
    attr() {} => {}
}

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(Active)]
//~^ ERROR cannot find value `foo` in this scope
//~| ERROR no rules expected `#`
#[forward_referenced_attr]
fn foo() {}
//~^ ERROR only builtin and autodiff attributes are supported on autodiff functions

fn main(){}

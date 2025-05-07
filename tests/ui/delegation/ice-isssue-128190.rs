#![feature(fn_delegation)]
#![allow(incomplete_features)]

fn a(&self) {}
//~^ ERROR `self` parameter is only allowed in associated functions

reuse a as b;

fn main() {}

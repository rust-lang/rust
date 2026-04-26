#![feature(fn_delegation)]

fn a(&self) {}
//~^ ERROR `self` parameter is only allowed in associated functions

reuse a as b;

fn main() {}

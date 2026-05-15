//@ compile-flags: -Z deduplicate-diagnostics=yes

#![feature(fn_delegation)]

reuse Default::default;
//~^ ERROR: delegation self type is not specified

fn main() {}

//@ compile-flags: -Z deduplicate-diagnostics=yes

#![feature(const_trait_impl)]
#![feature(fn_delegation)]

reuse Default::default;
//~^ ERROR: the trait bound `Self: [const] Default` is not satisfied

fn main() {}

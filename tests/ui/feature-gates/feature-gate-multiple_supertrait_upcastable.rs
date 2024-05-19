//@ check-pass

#![deny(multiple_supertrait_upcastable)]
//~^ WARNING unknown lint: `multiple_supertrait_upcastable`
#![warn(multiple_supertrait_upcastable)]
//~^ WARNING unknown lint: `multiple_supertrait_upcastable`

fn main() {}

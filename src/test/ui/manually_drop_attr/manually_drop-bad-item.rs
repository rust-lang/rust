#![feature(manually_drop_attr)]
#![forbid(unused_attributes)]

#[manually_drop]
//~^ ERROR attribute should be applied to a struct or enum
fn foo() {}

fn main() {}

#![feature(manually_drop_attr)]
#![forbid(unused_attributes)]
#![manually_drop]
//~^ ERROR `manually_drop` attribute cannot be used at crate level
//~^^ ERROR attribute should be applied to a struct or enum

#[manually_drop]
//~^ ERROR attribute should be applied to a struct or enum
fn foo() {}

fn main() {}

#![feature(trait_alias, marker_trait_attr)]

#[marker]
//~^ ERROR attribute cannot be used on
trait Foo = Send;

fn main() {}

#![feature(trait_alias, marker_trait_attr)]

#[marker]
//~^ ERROR attribute should be applied to a trait
trait Foo = Send;

fn main() {}

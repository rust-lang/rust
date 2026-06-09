#![deny(unused_attributes)]

#[doc(alias = "A")]
#[doc(alias = "A")] //~ ERROR
#[doc(alias = "B")]
#[doc(alias("B"))] //~ ERROR
pub struct Foo;

fn main() {}

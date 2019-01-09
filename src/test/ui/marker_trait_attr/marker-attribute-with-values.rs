#![feature(marker_trait_attr)]
#![feature(unrestricted_attribute_tokens)]

#[marker(always)]
trait Marker1 {}
//~^^ ERROR attribute should be empty

#[marker("never")]
trait Marker2 {}
//~^^ ERROR attribute should be empty

#[marker(key = value)]
trait Marker3 {}
//~^^ ERROR attribute should be empty

fn main() {}

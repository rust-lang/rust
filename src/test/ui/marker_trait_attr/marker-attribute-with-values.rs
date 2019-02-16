#![feature(marker_trait_attr)]
#![feature(unrestricted_attribute_tokens)]

#[marker(always)]
trait Marker1 {}
//~^^ ERROR attribute must be of the form

#[marker("never")]
trait Marker2 {}
//~^^ ERROR attribute must be of the form

#[marker(key = value)]
trait Marker3 {}
//~^^ ERROR expected unsuffixed literal or identifier, found value

fn main() {}

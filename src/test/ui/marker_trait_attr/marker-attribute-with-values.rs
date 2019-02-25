#![feature(marker_trait_attr)]

#[marker(always)]
trait Marker1 {}
//~^^ ERROR attribute must be of the form

#[marker("never")]
trait Marker2 {}
//~^^ ERROR attribute must be of the form

#[marker(key = "value")]
trait Marker3 {}
//~^^ ERROR attribute must be of the form `#[marker]`

fn main() {}

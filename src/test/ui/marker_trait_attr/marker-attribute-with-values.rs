#![feature(marker_trait_attr)]

#[marker(always)] //~ ERROR malformed `marker` attribute
trait Marker1 {}

#[marker("never")] //~ ERROR malformed `marker` attribute
trait Marker2 {}

#[marker(key = "value")] //~ ERROR malformed `marker` attribute
trait Marker3 {}

fn main() {}

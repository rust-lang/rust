#![feature(rustc_attrs)]

trait Foo {}

#[rustc_on_unimplemented] //~ ERROR malformed `rustc_on_unimplemented` attribute input
impl Foo for u32 {}

fn main() {}

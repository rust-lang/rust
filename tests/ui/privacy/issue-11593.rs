//@ aux-build:private-trait-xc.rs

extern crate private_trait_xc;

struct Bar;

impl private_trait_xc::Foo for Bar {}
//~^ ERROR: trait `Foo` is private

fn main() {}

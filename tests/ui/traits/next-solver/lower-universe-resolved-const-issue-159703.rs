//@ compile-flags: -Znext-solver=globally
//@ edition: 2015

#![allow(bare_trait_objects)]

trait Foo {}

struct BarType<const N: usize>;

impl<const N: usize> Foo for BarType {}
//~^ ERROR missing generics for struct `BarType`

fn a(x: &Foo) {
    let bar = BarType;
    a(bar);
    //~^ ERROR mismatched types
}

fn main() {}

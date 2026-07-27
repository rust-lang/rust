//@ compile-flags: -Znext-solver=globally

trait Foo {}

struct BarType<const N: usize>;

impl<const N: usize> Foo for BarType {}
//~^ ERROR missing generics for struct `BarType`

fn a(x: &dyn Foo) {
    let bar = BarType;
    a(bar);
    //~^ ERROR mismatched types
}

fn main() {}

trait Foo {}

trait Bar: Foo {}

impl Foo for () {}

impl Bar for () {}

fn main() {
    let bar: &dyn Bar = &();
    let foo: &dyn Foo = bar;
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected type `&dyn Foo`
    //~| NOTE expected trait `Foo`, found trait `Bar`
    //~| NOTE add `#![feature(trait_upcasting)]` to the crate attributes to enable
}

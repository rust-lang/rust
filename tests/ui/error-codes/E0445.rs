#![feature(type_privacy_lints)]
#[warn(private_bounds)]
#[warn(private_interfaces)]

// In this test both old and new private-in-public diagnostic were emitted.
// Old diagnostic will be deleted soon.
// See https://rust-lang.github.io/rfcs/2145-type-privacy.html.

trait Foo {
    fn dummy(&self) { }
}

pub trait Bar : Foo {}
//~^ ERROR private trait `Foo` in public interface [E0445]
//~| WARNING trait `Foo` is more private than the item `Bar`
pub struct Bar2<T: Foo>(pub T);
//~^ ERROR private trait `Foo` in public interface [E0445]
//~| WARNING trait `Foo` is more private than the item `Bar2`
pub fn foo<T: Foo> (t: T) {}
//~^ ERROR private trait `Foo` in public interface [E0445]
//~| WARNING trait `Foo` is more private than the item `foo`

fn main() {}

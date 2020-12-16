// This tests issue #79683: note in the error message that the trait is
// explicitely unimplemented instead of suggesting to implement it.

#![feature(negative_impls)]

struct Qux;
//~^ NOTE method `clone` not found for this
//~^^ NOTE method `foo` not found for this

impl !Clone for Qux {}

trait Bar {
    fn bar(&self);
}

impl !Bar for u32 {}

trait Foo {
    fn foo(&self);
}
//~^^^ NOTE `Foo` defines an item `foo`, perhaps you need to implement it

trait FooBar {
    fn foo(&self);
}

impl !Foo for Qux {}

impl !FooBar for Qux {}

impl !FooBar for u32 {}

fn main() {
    Qux.clone();
    //~^ ERROR no method named `clone` found for struct `Qux`
    //~| NOTE method not found in `Qux`
    //~| NOTE `Clone` defines an item `clone`, but is explicitely unimplemented

    0_u32.bar();
    //~^ ERROR no method named `bar` found for type `u32`
    //~| NOTE method not found in `u32`
    //~| NOTE `Bar` defines an item `bar`, but is explicitely unimplemented

    Qux.foo();
    //~^ ERROR no method named `foo` found for struct `Qux`
    //~| NOTE method not found in `Qux`
    //~| NOTE the following traits define an item `foo`, but are explicitely unimplemented

    0_u32.foo();
    //~^ ERROR no method named `foo` found for type `u32`
    //~| NOTE method not found in `u32`
    //~| NOTE `FooBar` defines an item `foo`, but is explicitely unimplemented
}

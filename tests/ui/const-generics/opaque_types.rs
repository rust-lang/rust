#![feature(type_alias_impl_trait)]

type Foo = impl Sized;
//~^ ERROR: cycle
//~| ERROR: cycle

fn foo<const C: Foo>() {}
//~^ ERROR: `Foo` is forbidden as the type of a const generic parameter
//~| ERROR: item does not constrain

fn main() {
    foo::<42>();
    //~^ ERROR: mismatched types
}

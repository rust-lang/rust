//@compile-flags: --test

#![warn(clippy::module_name_repetitions)]
#![allow(dead_code)]

pub mod foo {
    pub fn foo() {}
    pub fn foo_bar() {}
    //~^ ERROR: item name starts with its containing module's name
    //~| NOTE: `-D clippy::module-name-repetitions` implied by `-D warnings`
    pub fn bar_foo() {}
    //~^ ERROR: item name ends with its containing module's name
    pub struct FooCake;
    //~^ ERROR: item name starts with its containing module's name
    pub enum CakeFoo {}
    //~^ ERROR: item name ends with its containing module's name
    pub struct Foo7Bar;
    //~^ ERROR: item name starts with its containing module's name

    // Should not warn
    pub struct Foobar;

    // #8524 - shouldn't warn when item is declared in a private module...
    mod error {
        pub struct Error;
        pub struct FooError;
    }
    pub use error::Error;
    // ... but should still warn when the item is reexported to create a *public* path with repetition.
    pub use error::FooError;
    //~^ ERROR: item name starts with its containing module's name

    // FIXME: This should also warn because it creates the public path `foo::FooIter`.
    mod iter {
        pub struct FooIter;
    }
    pub use iter::*;

    // #12544 - shouldn't warn if item name consists only of an allowed prefix and a module name.
    pub fn to_foo() {}
    pub fn into_foo() {}
    pub fn as_foo() {}
    pub fn from_foo() {}
    pub fn try_into_foo() {}
    pub fn try_from_foo() {}
    pub trait IntoFoo {}
    pub trait ToFoo {}
    pub trait AsFoo {}
    pub trait FromFoo {}
    pub trait TryIntoFoo {}
    pub trait TryFromFoo {}
}

fn main() {}

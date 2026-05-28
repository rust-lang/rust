//@compile-flags: --test

#![warn(clippy::module_name_repetitions)]
#![allow(dead_code)]

pub mod foo {
    pub fn foo() {}
    pub fn foo_bar() {}
    //~^ module_name_repetitions

    pub fn bar_foo() {}
    //~^ module_name_repetitions

    pub struct FooCake;
    //~^ module_name_repetitions

    pub enum CakeFoo {}
    //~^ module_name_repetitions

    pub struct Foo7Bar;
    //~^ module_name_repetitions

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
    //~^ module_name_repetitions

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

pub mod issue14095 {
    pub mod widget {
        #[macro_export]
        macro_rules! define_widget {
            ($id:ident) => {
                /* ... */
            };
        }

        #[macro_export]
        macro_rules! widget_impl {
            ($id:ident) => {
                /* ... */
            };
        }
    }
}

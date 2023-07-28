//@compile-flags: --test

#![warn(clippy::module_name_repetitions)]
#![allow(dead_code)]

mod foo {
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
}

fn main() {}

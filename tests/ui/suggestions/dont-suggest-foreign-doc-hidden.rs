//@ edition:2015
//@ aux-build:hidden-struct.rs

extern crate hidden_struct;

#[doc(hidden)]
mod local {
    pub struct Foo;
}

pub fn test(_: Foo) {}
//~^ ERROR [E0425]

pub fn test2(_: Bar) {}
//~^ ERROR [E0425]

pub fn test3(_: Baz) {}
//~^ ERROR [E0425]

pub fn test4(_: Quux) {}
//~^ ERROR [E0425]

fn test5<T: hidden_struct::Marker>() {}

fn test6<T: hidden_struct::library::Marker>() {}

fn main() {
    test5::<i32>();
    //~^ ERROR [E0277]

    test6::<i32>();
    //~^ ERROR [E0277]
}

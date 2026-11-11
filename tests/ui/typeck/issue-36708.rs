//@ aux-build:issue-36708.rs

extern crate issue_36708 as lib;

struct Bar;

impl lib::Foo for Bar {
    fn foo<T>() {}
    //~^ ERROR E0049
}

fn main() {}

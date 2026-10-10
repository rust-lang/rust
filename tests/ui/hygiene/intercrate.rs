//@ aux-build:intercrate.rs

#![feature(decl_macro)]

extern crate intercrate;

fn main() {
    assert_eq!(intercrate::foo::m!(), 1);
    //~^ ERROR function `foo::bar::f` is private
}

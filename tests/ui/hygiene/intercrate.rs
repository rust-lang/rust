//@ aux-build:intercrate.rs

#![feature(decl_macro)]

extern crate intercrate;

fn main() {
    assert_eq!(intercrate::foo::m!(), 1);
    //~^ ERROR type `fn() -> u32 {foo::bar::f}` is private
}

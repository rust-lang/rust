#![deny(unused_attributes)]
#![feature(rustc_attrs)]

#![rustc_dummy] //~ ERROR unused attribute

#[rustc_dummy] //~ ERROR unused attribute
extern crate core;

#[rustc_dummy] //~ ERROR unused attribute
use std::collections;

#[rustc_dummy] //~ ERROR unused attribute
extern "C" {
    #[rustc_dummy] //~ ERROR unused attribute
    fn foo();
}

#[rustc_dummy] //~ ERROR unused attribute
mod foo {
    #[rustc_dummy] //~ ERROR unused attribute
    pub enum Foo {
        #[rustc_dummy] //~ ERROR unused attribute
        Bar,
    }
}

#[rustc_dummy] //~ ERROR unused attribute
fn bar(f: foo::Foo) {
    match f {
        #[rustc_dummy] //~ ERROR unused attribute
        foo::Foo::Bar => {}
    }
}

#[rustc_dummy] //~ ERROR unused attribute
struct Foo {
    #[rustc_dummy] //~ ERROR unused attribute
    a: isize
}

#[rustc_dummy] //~ ERROR unused attribute
trait Baz {
    #[rustc_dummy] //~ ERROR unused attribute
    fn blah(&self);
    #[rustc_dummy] //~ ERROR unused attribute
    fn blah2(&self) {}
}

fn main() {}

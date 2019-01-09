#![deny(unused_attributes)]
#![allow(dead_code, unused_imports, unused_extern_crates)]
#![feature(custom_attribute)]

#![foo] //~ ERROR unused attribute

#[foo] //~ ERROR unused attribute
extern crate core;

#[foo] //~ ERROR unused attribute
use std::collections;

#[foo] //~ ERROR unused attribute
extern "C" {
    #[foo] //~ ERROR unused attribute
    fn foo();
}

#[foo] //~ ERROR unused attribute
mod foo {
    #[foo] //~ ERROR unused attribute
    pub enum Foo {
        #[foo] //~ ERROR unused attribute
        Bar,
    }
}

#[foo] //~ ERROR unused attribute
fn bar(f: foo::Foo) {
    match f {
        #[foo] //~ ERROR unused attribute
        foo::Foo::Bar => {}
    }
}

#[foo] //~ ERROR unused attribute
struct Foo {
    #[foo] //~ ERROR unused attribute
    a: isize
}

#[foo] //~ ERROR unused attribute
trait Baz {
    #[foo] //~ ERROR unused attribute
    fn blah(&self);
    #[foo] //~ ERROR unused attribute
    fn blah2(&self) {}
}

fn main() {}

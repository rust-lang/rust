#![feature(arbitrary_self_types)]

use std::rc::Rc;

struct Foo;

impl Foo {
    fn foo(self: Rc<Self>) {} //~ ERROR arbitrary self types are only allowed for trait methods
}

fn main() {}

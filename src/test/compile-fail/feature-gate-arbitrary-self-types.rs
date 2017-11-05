use std::rc::Rc;

trait Foo {
    fn foo(self: Rc<Box<Self>>); //~ ERROR arbitrary `self` types are unstable
}

struct Bar;

impl Foo for Bar {
    fn foo(self: Rc<Box<Self>>) {} //~ ERROR arbitrary `self` types are unstable
}

impl Bar {
    fn bar(self: Box<Rc<Self>>) {} //~ ERROR arbitrary `self` types are unstable
}

fn main() {}

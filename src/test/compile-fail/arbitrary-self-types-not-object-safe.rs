#![feature(arbitrary_self_types)]

use std::rc::Rc;

trait Foo {
    fn foo(self: Rc<Self>) -> usize;
}

trait Bar {
    fn foo(self: Rc<Self>) -> usize where Self: Sized;
    fn bar(self: Box<Self>) -> usize;
}

impl Foo for usize {
    fn foo(self: Rc<Self>) -> usize {
        *self
    }
}

impl Bar for usize {
    fn foo(self: Rc<Self>) -> usize {
        *self
    }

    fn bar(self: Box<Self>) -> usize {
        *self
    }
}

fn make_foo() {
    let x = Box::new(5usize) as Box<Foo>;
    //~^ ERROR E0038
    //~| NOTE the method `foo` has an arbitrary self type
    //~| NOTE the trait `Foo` cannot be made into an object
}

fn make_bar() {
    let x = Box::new(5usize) as Box<Bar>;
    x.bar();
}

fn main() {}

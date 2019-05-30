#![feature(box_syntax)]

use std::cell::RefCell;
use std::rc::Rc;

trait Foo {
    fn set(&mut self, v: Rc<RefCell<A>>);
}

struct B {
    v: Option<Rc<RefCell<A>>>
}

impl Foo for B {
    fn set(&mut self, v: Rc<RefCell<A>>)
    {
        self.v = Some(v);
    }
}

struct A {
    v: Box<dyn Foo + Send>,
}

fn main() {
    let a = A {v: box B{v: None} as Box<dyn Foo + Send>};
    //~^ ERROR `std::rc::Rc<std::cell::RefCell<A>>` cannot be sent between threads safely
}

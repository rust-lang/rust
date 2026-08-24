#![feature(fn_delegation)]

use std::rc::Rc;

struct X;
impl X {
    fn foo(self: Rc<Box<Self>>, other: Box<Rc<Self>>) -> Option<Box<Self>> {
        None
    }
}

trait Trait {
    reuse X::foo;
    //~^ ERROR: cannot find function `foo` in `X`
}

reuse X::foo;
//~^ ERROR: cannot find function `foo` in `X`

fn main() {}

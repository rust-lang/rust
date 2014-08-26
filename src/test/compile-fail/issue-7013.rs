// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
    v: Box<Foo + Send>,
}

fn main() {
    let a = A {v: box B{v: None} as Box<Foo+Send>};
    //~^ ERROR cannot pack type `Box<B>`, which does not fulfill `Send`, as a trait bounded by Send
    let v = Rc::new(RefCell::new(a));
    let w = v.clone();
    let b = &*v;
    let mut b = b.borrow_mut();
    b.v.set(w.clone());
}

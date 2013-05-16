// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that rvalue lifetimes is limited to the enclosing trans
// cleanup scope. It is unclear that this is the correct lifetime for
// rvalues, but that's what it is right now.

struct Counter {
    value: uint
}

impl Counter {
    fn new(v: uint) -> Counter {
        Counter {value: v}
    }

    fn inc<'a>(&'a mut self) -> &'a mut Counter {
        self.value += 1;
        self
    }

    fn get(&self) -> uint {
        self.value
    }
}

pub fn main() {
    let v = Counter::new(22).inc().inc().get();
    //~^ ERROR borrowed value does not live long enough
    assert_eq!(v, 24);;
}

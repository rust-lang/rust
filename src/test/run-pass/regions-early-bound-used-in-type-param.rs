// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that you can use a fn lifetime parameter as part of
// the value for a type parameter in a bound.

trait Get<T> {
    fn get(&self) -> T;
}

struct Box<T> {
    t: T
}

impl<T:Clone> Get<T> for Box<T> {
    fn get(&self) -> T {
        self.t.clone()
    }
}

fn add<'a,G:Get<&'a int>>(g1: G, g2: G) -> int {
    *g1.get() + *g2.get()
}

pub fn main() {
    let b1 = Box { t: &3i };
    assert_eq!(add(b1, b1), 6i);
}

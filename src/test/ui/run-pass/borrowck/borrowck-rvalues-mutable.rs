// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct Counter {
    value: usize
}

impl Counter {
    fn new(v: usize) -> Counter {
        Counter {value: v}
    }

    fn inc<'a>(&'a mut self) -> &'a mut Counter {
        self.value += 1;
        self
    }

    fn get(&self) -> usize {
        self.value
    }

    fn get_and_inc(&mut self) -> usize {
        let v = self.value;
        self.value += 1;
        v
    }
}

pub fn main() {
    let v = Counter::new(22).get_and_inc();
    assert_eq!(v, 22);

    let v = Counter::new(22).inc().inc().get();
    assert_eq!(v, 24);
}

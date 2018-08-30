// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(arbitrary_self_types)]

use std::rc::Rc;

struct Foo<T: ?Sized>(T);

impl Foo<[u8]> {
    fn len(self: Rc<Self>) -> usize {
        self.0.len()
    }
}

fn main() {
    let rc = Rc::new(Foo([1u8,2,3])) as Rc<Foo<[u8]>>;
    assert_eq!(3, rc.len());
}

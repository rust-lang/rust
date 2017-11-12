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

trait Trait {
    fn trait_method<'a>(self: &'a Box<Rc<Self>>) -> &'a [i32];
}

impl Trait for Vec<i32> {
    fn trait_method<'a>(self: &'a Box<Rc<Self>>) -> &'a [i32] {
        &***self
    }
}

fn main() {
    let v = vec![1,2,3];

    assert_eq!(&[1,2,3], Box::new(Rc::new(v)).trait_method());
}

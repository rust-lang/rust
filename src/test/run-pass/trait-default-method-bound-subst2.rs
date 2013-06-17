// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(default_methods)];

trait A<T> {
    fn g(&self, x: T) -> T { x }
}

impl A<int> for int { }

fn f<T, V: A<T>>(i: V, j: T) -> T {
    i.g(j)
}

pub fn main () {
    assert_eq!(f(0, 2), 2);
}

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait Pushable<T> {
    fn push_val(&mut self, t: T);
}

impl<T> Pushable<T> for Vec<T> {
    fn push_val(&mut self, t: T) {
        self.push(t);
    }
}

pub fn main() {
    let mut v = vec!(1);
    v.push_val(2);
    v.push_val(3);
    assert_eq!(v, vec!(1, 2, 3));
}

// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Wrapper<'a, T: ?Sized>(&'a mut i32, T);

impl<'a, T: ?Sized> Drop for Wrapper<'a, T> {
    fn drop(&mut self) {
        *self.0 = 432;
    }
}

fn main() {
    let mut x = 0;
    {
        let wrapper = Box::new(Wrapper(&mut x, 123));
        let _: Box<Wrapper<Send>> = wrapper;
    }
    assert_eq!(432, x)
}

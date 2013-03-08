// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait foo {
    fn foo() -> uint;
}

impl<T> foo for ~[const T] {
    fn foo() -> uint { vec::len(self) }
}

pub fn main() {
    let v = ~[const 0];
    fail_unless!(v.foo() == 1u);
    let v = ~[0];
    fail_unless!(v.foo() == 1u);
    let mut v = ~[0];
    fail_unless!(v.foo() == 1u);
}

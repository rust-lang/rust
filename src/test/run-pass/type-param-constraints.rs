// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]


fn p_foo<T>(_pinned: T) { }
fn s_foo<T>(_shared: T) { }
fn u_foo<T:Send>(_unique: T) { }

struct r {
  i: int,
}

impl Drop for r {
    fn drop(&mut self) {}
}

fn r(i:int) -> r {
    r {
        i: i
    }
}

pub fn main() {
    p_foo(r(10));
    p_foo(@r(10));

    p_foo(box r(10));
    p_foo(@10);
    p_foo(box 10);
    p_foo(10);

    s_foo(@r(10));
    s_foo(@10);
    s_foo(box 10);
    s_foo(10);

    u_foo(box 10);
    u_foo(10);
}

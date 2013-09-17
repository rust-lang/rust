// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub static global: int = 3;

static global0: int = 4;
pub static global2: &'static int = &global0;

pub fn verify_same(a: &'static int) {
    let a = a as *int as uint;
    let b = &global as *int as uint;
    assert_eq!(a, b);
}

pub fn verify_same2(a: &'static int) {
    let a = a as *int as uint;
    let b = global2 as *int as uint;
    assert_eq!(a, b);
}

condition!{ pub test: int -> (); }

pub fn raise() {
    test::cond.raise(3);
}

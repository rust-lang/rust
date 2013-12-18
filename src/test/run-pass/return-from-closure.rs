// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// just to make sure that `return` is only returning from the closure,
// not the surrounding function.
static mut calls: uint = 0;

fn surrounding() {
    let return_works = |n: int| {
        unsafe { calls += 1 }

        if n >= 0 { return; }
        fail!()
    };

    return_works(10);
    return_works(20);


    let return_works_proc = proc(n: int) {
        unsafe { calls += 1 }

        if n >= 0 { return; }
        fail!()
    };

    return_works_proc(10);
}

pub fn main() {
    surrounding();

    assert_eq!(unsafe {calls}, 3);
}

// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "foo", kind = "static")]
extern {
    fn test_start(f: extern fn());
    fn test_end();
}

fn main() {
    unsafe {
        test_start(test_middle);
    }
}

struct A;

impl Drop for A {
    fn drop(&mut self) {
    }
}

extern fn test_middle() {
    let _a = A;
    foo();
}

fn foo() {
    let _a = A;
    unsafe {
        test_end();
    }
}

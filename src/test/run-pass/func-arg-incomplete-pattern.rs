// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do not leak when the arg pattern must drop part of the
// argument (in this case, the `y` field).


struct Foo {
    x: Box<uint>,
    y: Box<uint>,
}

fn foo(Foo {x, ..}: Foo) -> *const uint {
    let addr: *const uint = &*x;
    addr
}

pub fn main() {
    let obj = box 1;
    let objptr: *const uint = &*obj;
    let f = Foo {x: obj, y: box 2};
    let xptr = foo(f);
    assert_eq!(objptr, xptr);
}

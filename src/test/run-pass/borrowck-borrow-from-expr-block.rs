// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn borrow(x: &int, f: fn(x: &int)) {
    f(x)
}

fn test1(x: @~int) {
    // Right now, at least, this induces a copy of the unique pointer:
    do borrow({*x}) |p| {
        let x_a = ptr::addr_of(&(**x));
        assert (x_a as uint) != ptr::to_uint(p);
        assert unsafe{*x_a} == *p;
    }
}

fn main() {
    test1(@~22);
}

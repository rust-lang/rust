// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo(x: &[int]) -> int {
    x[0]
}

pub fn main() {
    let p = ~[1,2,3,4,5];
    let r = foo(p);
    assert!(r == 1);

    let p = ~[5,4,3,2,1];
    let r = foo(p);
    assert!(r == 5);
}

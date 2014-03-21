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
    let p = vec!(1,2,3,4,5);
    let r = foo(p.as_slice());
    assert_eq!(r, 1);

    let p = vec!(5,4,3,2,1);
    let r = foo(p.as_slice());
    assert_eq!(r, 5);
}

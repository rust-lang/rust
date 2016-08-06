// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the call operator autoderefs when calling a bounded type parameter.

use std::ops::FnMut;

fn call_with_2<F>(x: &mut F) -> isize
    where F : FnMut(isize) -> isize
{
    x(2) // look ma, no `*`
}

pub fn main() {
    let z = call_with_2(&mut |x| x - 22);
    assert_eq!(z, -20);
}

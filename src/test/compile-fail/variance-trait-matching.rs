// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #5781. Tests that subtyping is handled properly in trait matching.

trait Make<'a> {
    fn make(x: &'a mut isize) -> Self;
}

impl<'a> Make<'a> for &'a mut isize {
    fn make(x: &'a mut isize) -> &'a mut isize {
        x
    }
}

fn f() -> &'static mut isize {
    let mut x = 1;
    let y: &'static mut isize = Make::make(&mut x);   //~ ERROR `x` does not live long enough
    y
}

fn main() {}


// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests for the new |args| expr lambda syntax

fn f(i: int, f: &fn(int) -> int) -> int { f(i) }

fn g(g: &fn()) { }

fn ff() -> @fn(int) -> int {
    return |x| x + 1;
}

pub fn main() {
    assert!(f(10, |a| a) == 10);
    g(||());
    assert!(do f(10) |a| { a } == 10);
    do g() { }
    let _x: @fn() -> int = || 10;
    let _y: @fn(int) -> int = |a| a;
    assert!(ff()(10) == 11);
}

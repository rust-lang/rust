// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that two closures can simultaneously have immutable
// access to the variable, whether that immutable access be used
// for direct reads or for taking immutable ref. Also check
// that the main function can read the variable too while
// the closures are in scope. Issue #6801.

fn a() -> int {
    let mut x = 3;
    x += 1;
    let c1 = || x * 4;
    let c2 = || x * 5;
    c1() * c2() * x
}

fn get(x: &int) -> int {
    *x * 4
}

fn b() -> int {
    let mut x = 3;
    x += 1;
    let c1 = || get(&x);
    let c2 = || get(&x);
    c1() * c2() * x
}

fn c() -> int {
    let mut x = 3;
    x += 1;
    let c1 = || x * 5;
    let c2 = || get(&x);
    c1() * c2() * x
}

pub fn main() {
    assert_eq!(a(), 1280);
    assert_eq!(b(), 1024);
    assert_eq!(c(), 1280);
}

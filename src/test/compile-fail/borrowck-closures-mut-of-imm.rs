// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that two closures cannot simultaneously have mutable
// and immutable access to the variable. Issue #6801.

fn get(x: &int) -> int {
    *x
}

fn set(x: &mut int) {
    *x = 4;
}

fn a(x: &int) {
    let c1 = || set(&mut *x);
    //~^ ERROR cannot borrow
    let c2 = || set(&mut *x);
    //~^ ERROR cannot borrow
}

fn main() {
}

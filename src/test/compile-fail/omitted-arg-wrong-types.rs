// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test - #2093
fn let_in<T>(x: T, f: |T|) {}

fn main() {
    let_in(3u, |i| { assert!(i == 3); });
    //~^ ERROR expected `uint` but found `int`

    let_in(3, |i| { assert!(i == 3u); });
    //~^ ERROR expected `int` but found `uint`
}

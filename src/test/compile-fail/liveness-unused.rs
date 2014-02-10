// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
// xfail'd because lint is messed up with the new visitor transition

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deny(unused_variable)];
#[deny(dead_assignment)];

fn f1(x: int) {
    //~^ ERROR unused variable: `x`
}

fn f1b(x: &mut int) {
    //~^ ERROR unused variable: `x`
}

#[allow(unused_variable)]
fn f1c(x: int) {}

fn f2() {
    let x = 3;
    //~^ ERROR unused variable: `x`
}

fn f3() {
    let mut x = 3;
    //~^ ERROR variable `x` is assigned to, but never used
    x += 4;
    //~^ ERROR value assigned to `x` is never read
}

fn f3b() {
    let mut z = 3;
    //~^ ERROR variable `z` is assigned to, but never used
    loop {
        z += 4;
    }
}

#[allow(unused_variable)]
fn f3c() {
    let mut z = 3;
    loop { z += 4; }
}

#[allow(unused_variable)]
#[allow(dead_assignment)]
fn f3d() {
    let mut x = 3;
    x += 4;
}

fn f4() {
    match Some(3) {
      Some(i) => {
        //~^ ERROR unused variable: `i`
      }
      None => {}
    }
}

enum tri {
    a(int), b(int), c(int)
}

fn f4b() -> int {
    match a(3) {
      a(i) | b(i) | c(i) => {
        i
      }
    }
}

fn main() {
}

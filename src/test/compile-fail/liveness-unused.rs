// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn f1(x: int) {
    //~^ WARNING unused variable: `x`
}

fn f1b(x: &mut int) {
    //~^ WARNING unused variable: `x`
}

fn f2() {
    let x = 3;
    //~^ WARNING unused variable: `x`
}

fn f3() {
    let mut x = 3;
    //~^ WARNING variable `x` is assigned to, but never used
    x += 4;
    //~^ WARNING value assigned to `x` is never read
}

fn f3b() {
    let mut z = 3;
    //~^ WARNING variable `z` is assigned to, but never used
    loop {
        z += 4;
    }
}

fn f4() {
    match Some(3) {
      Some(i) => {
        //~^ WARNING unused variable: `i`
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

// leave this in here just to trigger compile-fail:
struct r {
    x: (),
}

impl r : Drop {
    fn finalize(&self) {}
}

fn main() {
    let x = r { x: () };
    fn@(move x) { copy x; }; //~ ERROR copying a noncopyable value
}

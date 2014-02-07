// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum opts {
    a(int), b(int), c(int)
}

fn matcher1(x: opts) {
    match x {
      a(ref i) | b(i) => {}
      //~^ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      c(_) => {}
    }
}

fn matcher2(x: opts) {
    match x {
      a(ref i) | b(i) => {}
      //~^ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      c(_) => {}
    }
}

fn matcher4(x: opts) {
    match x {
      a(ref mut i) | b(ref i) => {}
      //~^ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      c(_) => {}
    }
}


fn matcher5(x: opts) {
    match x {
      a(ref i) | b(ref i) => {}
      c(_) => {}
    }
}

fn main() {}

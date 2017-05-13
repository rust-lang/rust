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
    a(isize), b(isize), c(isize)
}

fn matcher1(x: opts) {
    match x {
      opts::a(ref i) | opts::b(i) => {}
      //~^ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      //~^^ ERROR mismatched types
      opts::c(_) => {}
    }
}

fn matcher2(x: opts) {
    match x {
      opts::a(ref i) | opts::b(i) => {}
      //~^ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      //~^^ ERROR mismatched types
      opts::c(_) => {}
    }
}

fn matcher4(x: opts) {
    match x {
      opts::a(ref mut i) | opts::b(ref i) => {}
      //~^ ERROR variable `i` is bound with different mode in pattern #2 than in pattern #1
      //~^^ ERROR mismatched types
      opts::c(_) => {}
    }
}


fn matcher5(x: opts) {
    match x {
      opts::a(ref i) | opts::b(ref i) => {}
      opts::c(_) => {}
    }
}

fn main() {}

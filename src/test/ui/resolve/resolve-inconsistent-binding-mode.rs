// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Opts {
    A(isize), B(isize), C(isize)
}

fn matcher1(x: Opts) {
    match x {
      Opts::A(ref i) | Opts::B(i) => {}
      //~^ ERROR variable `i` is bound in inconsistent ways within the same match arm
      //~^^ ERROR mismatched types
      Opts::C(_) => {}
    }
}

fn matcher2(x: Opts) {
    match x {
      Opts::A(ref i) | Opts::B(i) => {}
      //~^ ERROR variable `i` is bound in inconsistent ways within the same match arm
      //~^^ ERROR mismatched types
      Opts::C(_) => {}
    }
}

fn matcher4(x: Opts) {
    match x {
      Opts::A(ref mut i) | Opts::B(ref i) => {}
      //~^ ERROR variable `i` is bound in inconsistent ways within the same match arm
      //~^^ ERROR mismatched types
      Opts::C(_) => {}
    }
}


fn matcher5(x: Opts) {
    match x {
      Opts::A(ref i) | Opts::B(ref i) => {}
      Opts::C(_) => {}
    }
}

fn main() {}

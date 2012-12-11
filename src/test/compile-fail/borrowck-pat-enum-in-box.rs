// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn match_imm_box(v: &const @Option<int>) -> int {
    match *v {
      @Some(ref i) => {*i}
      @None => {0}
    }
}

fn match_const_box(v: &const @const Option<int>) -> int {
    match *v {
      @Some(ref i) => { *i } // ok because this is pure
      @None => {0}
    }
}

pure fn pure_process(_i: int) {}

fn match_const_box_and_do_pure_things(v: &const @const Option<int>) {
    match *v {
      @Some(ref i) => {
        pure_process(*i)
      }
      @None => {}
    }
}

fn process(_i: int) {}

fn match_const_box_and_do_bad_things(v: &const @const Option<int>) {
    match *v {
      @Some(ref i) => { //~ ERROR illegal borrow unless pure
        process(*i) //~ NOTE impure due to access to impure function
      }
      @None => {}
    }
}

fn main() {
}

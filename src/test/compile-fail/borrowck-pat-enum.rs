// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn match_ref(v: Option<int>) -> int {
    match v {
      Some(ref i) => {
        *i
      }
      None => {0}
    }
}

fn match_ref_unused(v: Option<int>) {
    match v {
      Some(_) => {}
      None => {}
    }
}

fn match_const_reg(v: &const Option<int>) -> int {
    match *v {
      Some(ref i) => {*i} //~ ERROR cannot borrow
        //~^ ERROR unsafe borrow
      None => {0}
    }
}

fn impure(_i: int) {
}

fn match_const_reg_unused(v: &const Option<int>) {
    match *v {
      Some(_) => {impure(0)} // OK because nothing is captured
      None => {}
    }
}

fn match_const_reg_impure(v: &const Option<int>) {
    match *v {
      Some(ref i) => {impure(*i)} //~ ERROR cannot borrow
        //~^ ERROR unsafe borrow
      None => {}
    }
}

fn match_imm_reg(v: &Option<int>) {
    match *v {
      Some(ref i) => {impure(*i)} // OK because immutable
      None => {}
    }
}

fn match_mut_reg(v: &mut Option<int>) {
    match *v {
      Some(ref i) => {impure(*i)} // OK, frozen
      None => {}
    }
}

fn main() {
}

// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty

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

fn impure(_i: int) {
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

pub fn main() {
}

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

#[derive(Copy, Clone)]
enum side { mayo, catsup, vinegar }
#[derive(Copy, Clone)]
enum order { hamburger, fries(side), shake }
#[derive(Copy, Clone)]
enum meal { to_go(order), for_here(order) }

fn foo(m: Box<meal>, cond: bool) {
    match *m {
      meal::to_go(_) => { }
      meal::for_here(_) if cond => {}
      meal::for_here(order::hamburger) => {}
      meal::for_here(order::fries(_s)) => {}
      meal::for_here(order::shake) => {}
    }
}

pub fn main() {
    foo(box meal::for_here(order::hamburger), true)
}

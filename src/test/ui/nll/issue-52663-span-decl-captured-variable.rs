// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

fn expect_fn<F>(f: F) where F : Fn() {
    f();
}

fn main() {
   {
       let x = (vec![22], vec![44]);
       expect_fn(|| drop(x.0));
       //~^ ERROR cannot move out of captured variable in an `Fn` closure [E0507]
   }
}

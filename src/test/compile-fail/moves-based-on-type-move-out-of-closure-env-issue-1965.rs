// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax, unboxed_closures)]

use std::usize;

fn to_fn<A,F:Fn<A>>(f: F) -> F { f }

fn test(_x: Box<usize>) {}

fn main() {
    let i = box 3;
    let _f = to_fn(|| test(i)); //~ ERROR cannot move out
}

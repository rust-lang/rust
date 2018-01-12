// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #38899

#![feature(nll)]
#![allow(dead_code)]

pub struct Block<'a> {
    current: &'a u8,
    unrelated: &'a u8,
}

fn bump<'a>(mut block: &mut Block<'a>) {
    let x = &mut block;
    println!("{}", x.current);
    let p: &'a u8 = &*block.current;
    //~^ ERROR cannot borrow `*block.current` as immutable because it is also borrowed as mutable
    drop(x);
    drop(p);
}

fn main() {}

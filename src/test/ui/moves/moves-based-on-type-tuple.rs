// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

// compile-flags: -Z borrowck=compare

fn dup(x: Box<isize>) -> Box<(Box<isize>,Box<isize>)> {
    box (x, x)
    //~^ use of moved value: `x` (Ast) [E0382]
    //~| use of moved value: `x` (Mir) [E0382]
}

fn main() {
    dup(box 3);
}

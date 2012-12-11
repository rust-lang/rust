// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn test1() {
    let v: int;
    let mut w: int;
    v = 1; //~ NOTE prior assignment occurs here
    w = 2;
    v <-> w; //~ ERROR re-assignment of immutable variable
}

fn test2() {
    let v: int;
    let mut w: int;
    v = 1; //~ NOTE prior assignment occurs here
    w = 2;
    w <-> v; //~ ERROR re-assignment of immutable variable
}

fn main() {
}

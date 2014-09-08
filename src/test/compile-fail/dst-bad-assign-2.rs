// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Forbid assignment into a dynamically sized type.

struct Fat<Sized? T> {
    f1: int,
    f2: &'static str,
    ptr: T
}

#[deriving(PartialEq,Eq)]
struct Bar;

#[deriving(PartialEq,Eq)]
struct Bar1 {
    f: int
}

trait ToBar {
    fn to_bar(&self) -> Bar;
    fn to_val(&self) -> int;
}

impl ToBar for Bar1 {
    fn to_bar(&self) -> Bar {
        Bar
    }
    fn to_val(&self) -> int {
        self.f
    }
}

pub fn main() {
    // Assignment.
    let f5: &mut Fat<ToBar> = &mut Fat { f1: 5, f2: "some str", ptr: Bar1 {f :42} };
    let z: Box<ToBar> = box Bar1 {f: 36};
    f5.ptr = *z;  //~ ERROR dynamically sized type on lhs of assignment
    //~^ ERROR E0161
}

// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test privacy and struct inheritance.
#![feature(struct_inherit)]

mod Foo {
    pub virtual struct S1 {
        pub f1: int,
        f2: int,
    }
}

struct S2 : Foo::S1 {
    pub f3: int,
}

impl S2 {
    fn new() -> S2 {
        S2{f1: 3, f2: 4, f3: 5} //~ ERROR field `f2` of struct `S2` is private
    }

    fn bar(&self) {
        self.f3;
        self.f1;
        self.f2; //~ ERROR field `f2` of struct `S2` is private
    }
}

pub fn main() {
    let s = S2{f1: 3, f2: 4, f3: 5}; //~ ERROR field `f2` of struct `S2` is private
    s.f3;
    s.f1;
    s.f2; //~ ERROR field `f2` of struct `S2` is private
}

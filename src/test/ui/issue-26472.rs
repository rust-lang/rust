// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod sub {
    pub struct S { len: usize }
    impl S {
        pub fn new() -> S { S { len: 0 } }
        pub fn len(&self) -> usize { self.len }
    }
}

fn main() {
    let s = sub::S::new();
    let v = s.len;
    //~^ ERROR field `len` of struct `sub::S` is private
    //~| NOTE a method `len` also exists, perhaps you wish to call it
}

// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo<'a> {
    a: &'a i32,
}

impl<'a> Foo<'a> {
   //~^  NOTE first declared here
    fn f<'a>(x: &'a i32) { //~ ERROR E0496
       //~^ NOTE lifetime 'a already in scope
    }
}

fn main() {
}

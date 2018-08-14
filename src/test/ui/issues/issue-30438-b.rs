// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Modified regression test for Issue #30438 that exposed an
// independent issue (see discussion on ticket).

use std::ops::Index;

struct Test<'a> {
    s: &'a String
}

impl <'a> Index<usize> for Test<'a> {
    type Output = Test<'a>;
    fn index(&self, _: usize) -> &Self::Output {
        &Test { s: &self.s}
        //~^ ERROR: borrowed value does not live long enough
    }
}

fn main() {
    let s = "Hello World".to_string();
    let test = Test{s: &s};
    let r = &test[0];
    println!("{}", test.s); // OK since test is valid
    println!("{}", r.s); // Segfault since value pointed by r has already been dropped
}

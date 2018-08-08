// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test the case where the `Self` type has a bound lifetime that must
// be adjusted in the fn signature. Issue #19537.

use std::collections::HashMap;

struct Foo<'a> {
    map: HashMap<usize, &'a str>
}

impl<'a> Foo<'a> {
    fn new() -> Foo<'a> { panic!() }
    fn insert(&'a mut self) { }
}
fn main() {
    let mut foo = Foo::new();
    foo.insert();
    foo.insert(); //~ ERROR cannot borrow
}

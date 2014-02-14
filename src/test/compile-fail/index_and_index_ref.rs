// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    bar: Bar
}

#[deriving(Clone)]
struct Bar {
    age: int
}

impl Index<int,Bar> for Foo {
    fn index(&self, element: &int) -> Bar {
        self.bar.clone()
    }
}

impl IndexRef<int,Bar> for Foo {
    fn index<'a>(&'a self, element: &int) -> &'a Bar {
        &self.bar
    }
}

fn main() {
  let foo = Foo{ bar: Bar{ age: 50 } };
  foo[10]; //~ ERROR multiple applicable methods in scope
}

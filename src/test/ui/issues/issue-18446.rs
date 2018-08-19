// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that name clashes between the method in an impl for the type
// and the method in the trait when both are in the same scope.

trait T {
    fn foo(&self);
}

impl<'a> T + 'a {
    fn foo(&self) {}
}

impl T for i32 {
    fn foo(&self) {}
}

fn main() {
    let x: &T = &0i32;
    x.foo(); //~ ERROR multiple applicable items in scope [E0034]
}

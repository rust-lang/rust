// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a method call where the parameter `B` would (illegally) be
// inferred to a region bound in the method argument. If this program
// were accepted, then the closure passed to `s.f` could escape its
// argument.

struct S;

impl S {
    fn f<B, F>(&self, _: F) where F: FnOnce(&i32) -> B {
    }
}

fn main() {
    let s = S;
    s.f(|p| p) //~ ERROR cannot infer
}

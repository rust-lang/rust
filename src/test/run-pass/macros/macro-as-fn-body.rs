// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// run-pass
//
// Description - ensure Interpolated blocks can act as valid function bodies
// Covered cases: free functions, struct methods, and default trait functions

macro_rules! def_fn {
    ($body:block) => {
        fn bar() $body
    }
}

trait Foo {
    def_fn!({ println!("foo"); });
}

struct Baz {}

impl Foo for Baz {}

struct Qux {}

impl Qux {
    def_fn!({ println!("qux"); });
}

def_fn!({ println!("quux"); });

pub fn main() {
    Baz::bar();
    Qux::bar();
    bar();
}

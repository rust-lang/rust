// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that duplicate methods in impls are not allowed

struct Foo;

trait Bar {
    fn bar(&self) -> int;
}

impl Bar for Foo {
    fn bar(&self) -> int {1}
    fn bar(&self) -> int {2} //~ ERROR duplicate method
}

fn main() {
    println!("{}", Foo.bar());
}

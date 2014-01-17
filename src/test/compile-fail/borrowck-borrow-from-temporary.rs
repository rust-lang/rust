// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test lifetimes are linked properly when we take reference
// to interior.

struct Foo(int);

fn foo() -> &int {
    let &Foo(ref x) = &Foo(3); //~ ERROR borrowed value does not live long enough
    x
}

pub fn main() {
}

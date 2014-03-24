// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that lifetimes can't escape through owned trait casts

trait X {}
impl<'a> X for &'a X {}

fn foo(x: &X) -> ~X: {
    ~x as ~X:
    //~^ ERROR lifetime of the source pointer does not outlive lifetime bound of the object type
}

fn main() {
}
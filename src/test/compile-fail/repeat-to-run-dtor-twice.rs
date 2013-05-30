// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that one can't run a destructor twice with the repeated vector
// literal syntax.

struct Foo {
    x: int,

}

impl Drop for Foo {
    fn finalize(&self) {
        println("Goodbye!");
    }
}

fn main() {
    let a = Foo { x: 3 };
    let _ = [ a, ..5 ];     //~ ERROR copying a value of non-copyable type
}

// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::Cell;

struct Foo {
    x: int
}

impl Clone for Foo {
    fn clone(&self) -> Foo {
        // Using Cell in any way should never cause clone() to be
        // invoked -- after all, that would permit evil user code to
        // abuse `Cell` and trigger crashes.

        fail!();
    }
}

pub fn main() {
    let x = Cell::new(Foo { x: 22 });
    let _y = x.get();
    let _z = x.clone();
}

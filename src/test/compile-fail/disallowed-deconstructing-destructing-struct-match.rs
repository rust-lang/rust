// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct X {
    x: ~str,
}

impl Drop for X {
    fn drop(&mut self) {
        error!("value: %s", self.x);
    }
}

fn main() {
    let x = X { x: ~"hello" };

    match x {
        X { x: y } => error!("contents: %s", y)
        //~^ ERROR cannot move out of type `X`, which defines the `Drop` trait
    }
}

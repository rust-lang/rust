// xfail-test #3024
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
    fn finalize(&self) {
        error!("value: %s", self.x);
    }
}

fn unwrap(x: X) -> ~str {
    let X { x: y } = x; //~ ERROR cannot bind by-move within struct
    y
}

fn main() {
    let x = X { x: ~"hello" };
    let y = unwrap(x);
    error!("contents: %s", y);
}

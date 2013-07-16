// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: instantiating a type parameter with an incompatible type

struct r {
  i:int
}

fn r(i:int) -> r { r { i: i } }

impl Drop for r {
    fn drop(&self) {}
}

fn main() {
    // This can't make sense as it would copy the classes
    let i = ~[r(0)];
    let j = ~[r(1)];
    let k = i + j;
    info!(j);
}

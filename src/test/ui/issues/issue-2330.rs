// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum chan { }

trait channel<T> {
    fn send(&self, v: T);
}

// `chan` is not a trait, it's an enum
impl chan for isize { //~ ERROR expected trait, found enum `chan`
    fn send(&self, v: isize) { panic!() }
}

fn main() {
}

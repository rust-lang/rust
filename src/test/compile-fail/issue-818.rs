// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod ctr {

    pub enum ctr { priv mkCtr(int) }

    pub fn new(i: int) -> ctr { mkCtr(i) }
    pub fn inc(c: ctr) -> ctr { mkCtr(*c + 1) }
}


fn main() {
    let c = ctr::new(42);
    let c2 = ctr::inc(c);
    assert!(*c2 == 5); //~ ERROR can only dereference enums with a single, public variant
}

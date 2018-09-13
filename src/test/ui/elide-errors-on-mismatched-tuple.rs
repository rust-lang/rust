// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Hide irrelevant E0277 errors (#50333)

trait T {}

struct A;
impl T for A {}
impl A {
    fn new() -> Self {
        Self {}
    }
}

fn main() {
    let (a, b, c) = (A::new(), A::new()); // This tuple is 2 elements, should be three
    //~^ ERROR mismatched types
    let ts: Vec<&T> = vec![&a, &b, &c];
    // There is no E0277 error above, as `a`, `b` and `c` are `TyErr`
}

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    // Note: explicit type annot is required here
    // because otherwise the inference gets smart
    // and assigns a type of ~[~[const int]].
    let mut v: ~[~[int]] = ~[~[0]];

    fn f(&&v: ~[~[const int]]) {
        v[0] = ~[3]
    }

    f(v); //~ ERROR (values differ in mutability)
}

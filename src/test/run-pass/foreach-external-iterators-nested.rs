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
    let x = [1,..100];
    let y = [2,..100];
    let mut p = 0;
    let mut q = 0;
    foreach i in x.iter() {
        foreach j in y.iter() {
            p += *j;
        }
        q += *i + p;
    }
    assert!(q == 1010100);
}

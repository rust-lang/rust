// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo(cond: bool) {
    let x = 5;
    let mut y: &blk/int = &x;

    let mut z: &blk/int;
    if cond {
        z = &x; //~ ERROR cannot infer an appropriate lifetime due to conflicting requirements
    } else {
        let w: &blk/int = &x;
        z = w;
    }
}

fn main() {
}

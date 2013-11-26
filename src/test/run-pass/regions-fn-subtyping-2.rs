// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #2263.

// Here, `f` is a function that takes a pointer `x` and a function
// `g`, where `g` requires its argument `y` to be in the same region
// that `x` is in.
fn has_same_region(f: <'a>|x: &'a int, g: |y: &'a int||) {
    // `f` should be the type that `wants_same_region` wants, but
    // right now the compiler complains that it isn't.
    wants_same_region(f);
}

fn wants_same_region(_f: <'b>|x: &'b int, g: |y: &'b int||) {
}

pub fn main() {
}

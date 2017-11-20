// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we do suggest `(ref a, ref b)` here, since `a` and `b`
// are nested within a pattern
fn main() {
    let x = vec![(String::new(), String::new())];
    let (a, b) = x[0]; //~ ERROR cannot move out of indexed content
}

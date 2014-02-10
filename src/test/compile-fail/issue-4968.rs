// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #4968

static A: (int,int) = (4,2);
fn main() {
    match 42 { A => () }
    //~^ ERROR mismatched types: expected `<generic integer #0>` but found `(int,int)`
    //         (expected integral variable but found tuple)
}

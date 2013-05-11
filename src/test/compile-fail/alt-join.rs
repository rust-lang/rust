// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// a good test that we merge paths correctly in the presence of a
// variable that's used before it's declared

fn my_fail() -> ! { fail!(); }

fn main() {
    match true { false => { my_fail(); } true => { } }

    debug!(x); //~ ERROR unresolved name `x`.
    let x: int;
}

// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for a specific corner case: when we compute the LUB of two fn
// types and their parameters have unbound variables. In that case, we
// wind up relating those two variables. This was causing an ICE in an
// in-progress PR.

fn main() {
    let a_f: fn(_) = |_| ();
    let b_f: fn(_) = |_| ();
    let c_f = match 22 {
        0 => a_f,
        _ => b_f,
    };
    c_f(4);
}

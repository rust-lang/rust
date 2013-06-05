// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32

use std::result;
use std::task;

fn adder(x: @int, y: @int) -> int { return *x + *y; }
fn failer() -> @int { fail!(); }
pub fn main() {
    assert!(result::is_err(&task::try(|| {
        adder(@2, failer()); ()
    })));
}

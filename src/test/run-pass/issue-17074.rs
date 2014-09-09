// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static X: u64 = -1 as u16 as u64;
static Y: u64 = -1 as u32 as u64;

fn main() {
    assert_eq!(match 1 {
        X => unreachable!(),
        Y => unreachable!(),
        _ => 1i
    }, 1);
}

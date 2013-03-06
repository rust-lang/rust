// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn inc(x: &uint) -> uint { *x + 1 }

pub fn main() {
    fail_unless!([1, 3].map_to_vec(inc) == ~[2, 4]);
    fail_unless!([1, 2, 3].map_to_vec(inc) == ~[2, 3, 4]);
    fail_unless!(iter::map_to_vec(&None::<uint>, inc) == ~[]);
    fail_unless!(iter::map_to_vec(&Some(1u), inc) == ~[2]);
    fail_unless!(iter::map_to_vec(&Some(2u), inc) == ~[3]);
}

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    fail_unless!([1u, 3u].to_vec() == ~[1u, 3u]);
    let e: ~[uint] = ~[];
    fail_unless!(e.to_vec() == ~[]);
    fail_unless!(iter::to_vec(&None::<uint>) == ~[]);
    fail_unless!(iter::to_vec(&Some(1u)) == ~[1u]);
    fail_unless!(iter::to_vec(&Some(2u)) == ~[2u]);
}

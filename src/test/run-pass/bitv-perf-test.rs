// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;
use extra::bitv::Bitv;

fn bitv_test() {
    let mut v1 = ~Bitv::new(31, false);
    let v2 = ~Bitv::new(31, true);
    v1.union(v2);
}

pub fn main() {
    do 10000.times || {bitv_test()};
}

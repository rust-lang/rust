// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test makes sure that we detect changed feature gates.

// revisions:rpass1 cfail2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![cfg_attr(rpass1, feature(nll))]

fn main() {
    let mut v = vec![1];
    v.push(v[0]);
    //[cfail2]~^ ERROR cannot borrow
}

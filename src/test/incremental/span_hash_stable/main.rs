// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test makes sure that it doesn't make a difference in which order we are
// adding source files to the codemap. The order affects the BytePos values of
// the spans and this test makes sure that we handle them correctly by hashing
// file:line:column instead of raw byte offset.

// revisions:rpass1 rpass2
// compile-flags: -g -Z query-dep-graph

#![feature(rustc_attrs)]

mod auxiliary;

fn main() {
    let _ = auxiliary::sub1::SomeType {
        x: 0,
        y: 1,
    };

    let _ = auxiliary::sub2::SomeOtherType {
        a: 2,
        b: 3,
    };
}


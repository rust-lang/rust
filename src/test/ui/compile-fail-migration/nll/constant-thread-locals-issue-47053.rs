// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #47053

#![feature(nll)]
#![feature(thread_local)]

#[thread_local]
static FOO: isize = 5;

fn main() {
    FOO = 6; //~ ERROR cannot assign to immutable static item `FOO` [E0594]
}

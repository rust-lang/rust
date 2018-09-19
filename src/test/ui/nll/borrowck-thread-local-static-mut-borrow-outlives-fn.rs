// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// run-pass
//
// FIXME(#54366) - We probably shouldn't allow #[thread_local] static mut to get a 'static lifetime.

#![feature(nll)]
#![feature(thread_local)]

#[thread_local]
static mut X1: u64 = 0;

struct S1 {
    a: &'static mut u64,
}

impl S1 {
    fn new(_x: u64) -> S1 {
        S1 {
            a: unsafe { &mut X1 },
        }
    }
}

fn main() {
    S1::new(0).a;
}

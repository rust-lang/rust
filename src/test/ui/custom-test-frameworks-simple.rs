// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --test
// run-pass

#![feature(custom_test_frameworks)]
#![test_runner(crate::foo_runner)]

#[cfg(test)]
fn foo_runner(ts: &[&Fn(usize)->()]) {
    for (i, t) in ts.iter().enumerate() {
        t(i);
    }
}

#[test_case]
fn test1(i: usize) {
    println!("Hi #{}", i);
}

#[test_case]
fn test2(i: usize) {
    println!("Hey #{}", i);
}

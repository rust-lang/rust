// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test closure that takes two references and is supposed to return
// the first, but actually returns the second. This should fail within
// the closure.

// compile-flags:-Znll -Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

#[rustc_regions]
fn test() {
    expect_sig(|a, b| b); // ought to return `a`
    //~^ WARN not reporting region error due to -Znll
    //~| ERROR does not outlive free region
}

fn expect_sig<F>(f: F) -> F
    where F: for<'a> FnMut(&'a i32, &i32) -> &'a i32
{
    f
}

fn deref(_p: &i32) { }

fn main() { }

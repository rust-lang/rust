// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #38714. This code uses some creepy code paths
// that (as of the time of this writing) mix-and-match the bound
// regions from the expected/supplied types when deciding on a closure
// signature.  Cleaning up those paths initially led to an ICE;
// reverting the relevant PR causes a proper error.

fn foo<F>(f: F) where F: for<'a> Fn(&'a str) -> &'a str {}
fn bar<F>(f: F) where F: Fn(&str) -> &str {}

fn main() {
    foo(|a: &str| a); // Compiler panic
    bar(|a: &str| a); // Works

    let local = |a: &str| a;
    bar(local);  //~ ERROR type mismatch
}

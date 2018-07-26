// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

fn foo<F: Fn(usize)>(_: F) {}
fn bar<F: Fn<usize>>(_: F) {}
fn main() {
    fn f(_: u64) {}
    foo(|_: isize| {}); //~ ERROR type mismatch
    bar(|_: isize| {}); //~ ERROR type mismatch
    foo(f); //~ ERROR type mismatch
    bar(f); //~ ERROR type mismatch
}

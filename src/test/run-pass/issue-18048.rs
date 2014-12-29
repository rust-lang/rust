// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-18048-lib.rs

// Test that we are able to reference cross-crate traits that employ
// associated types.

#![feature(associated_types)]

extern crate "issue-18048-lib" as bar;

use bar::Bar;

fn foo<B:Bar>(b: B) -> <B as Bar>::T {
    Bar::get(None::<B>)
}

fn main() {
    println!("{}", foo(3i));
}

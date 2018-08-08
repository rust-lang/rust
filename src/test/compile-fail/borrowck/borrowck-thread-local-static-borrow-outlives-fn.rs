// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

#![feature(thread_local)]

#[thread_local]
static FOO: u8 = 3;

fn assert_static(_t: &'static u8) {}
fn main() {
     assert_static(&FOO); //[ast]~ ERROR [E0597]
                          //[mir]~^ ERROR [E0712]
}

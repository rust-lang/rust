// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// regression test for #8005

macro_rules! test { () => { fn foo() -> i32 { 1; } } }
                                           //~^ ERROR mismatched types
                                           //~| HELP consider removing this semicolon

fn no_return() -> i32 {} //~ ERROR mismatched types

fn bar(x: u32) -> u32 { //~ ERROR mismatched types
    x * 2; //~ HELP consider removing this semicolon
}

fn baz(x: u64) -> u32 { //~ ERROR mismatched types
    x * 2;
}

fn main() {
    test!();
}

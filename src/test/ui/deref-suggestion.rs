// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! borrow {
    ($x:expr) => { &$x } //~ ERROR mismatched types
}

fn foo(_: String) {}

fn foo2(s: &String) {
    foo(s); //~ ERROR mismatched types
}

fn foo3(_: u32) {}
fn foo4(u: &u32) {
    foo3(u); //~ ERROR mismatched types
}

fn main() {
    let s = String::new();
    let r_s = &s;
    foo2(r_s);
    foo(&"aaa".to_owned()); //~ ERROR mismatched types
    foo(&mut "aaa".to_owned()); //~ ERROR mismatched types
    foo3(borrow!(0));
    foo4(&0);
}

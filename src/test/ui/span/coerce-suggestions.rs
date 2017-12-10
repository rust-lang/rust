// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

fn test(_x: &mut String) {}
fn test2(_x: &mut i32) {}

fn main() {
    let x: usize = String::new();
    //~^ ERROR E0308
    let x: &str = String::new();
    //~^ ERROR E0308
    let y = String::new();
    test(&y);
    //~^ ERROR E0308
    test2(&y);
    //~^ ERROR E0308
    let f;
    f = box f;
    //~^ ERROR E0308

    let s = &mut String::new();
    s = format!("foo");
}

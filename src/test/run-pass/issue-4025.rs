// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
# if b { x } else { y } requires identical types for x and y
*/

fn print1(b: bool, s1: &str, s2: &str) {
    println(if b { s1 } else { s2 });
}
fn print2<'a, 'b>(b: bool, s1: &'a str, s2: &'b str) {
    println(if b { s1 } else { s2 });
}
fn print3(b: bool, s1: &str, s2: &str) {
    let mut s: &str;
    if b { s = s1; } else { s = s2; }
    println(s);
}
fn print4<'a, 'b>(b: bool, s1: &'a str, s2: &'b str) {
    let mut s: &str;
    if b { s = s1; } else { s = s2; }
    println(s);
}

pub fn main() {}

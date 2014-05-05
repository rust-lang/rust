// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

// compile-flags: -D type-limits
fn main() { }

fn foo() {
    let mut i = 100u;
    while i >= 0 { //~ ERROR comparison is useless due to type limits
        i -= 1;
    }
}

fn bar() -> i8 {
    return 123;
}

fn baz() -> bool {
    128 > bar() //~ ERROR comparison is useless due to type limits
                //~^ WARNING literal out of range for its type
}

fn bleh() {
    let u = 42u8;
    let _ = u > 255; //~ ERROR comparison is useless due to type limits
    let _ = 255 < u; //~ ERROR comparison is useless due to type limits
    let _ = u < 0; //~ ERROR comparison is useless due to type limits
    let _ = 0 > u; //~ ERROR comparison is useless due to type limits
    let _ = u <= 255; //~ ERROR comparison is useless due to type limits
    let _ = 255 >= u; //~ ERROR comparison is useless due to type limits
    let _ = u >= 0; //~ ERROR comparison is useless due to type limits
    let _ = 0 <= u; //~ ERROR comparison is useless due to type limits
}

fn qux() {
    let mut i = 1i8;
    while 200 != i { //~ ERROR comparison is useless due to type limits
                     //~^ WARNING literal out of range for its type
        i += 1;
    }
}

fn quy() {
    let i = -23u; //~ WARNING negation of unsigned int literal may be unintentional
                  //~^ WARNING unused variable
}

fn quz() {
    let i = 23u;
    let j = -i;   //~ WARNING negation of unsigned int variable may be unintentional
                  //~^ WARNING unused variable
}

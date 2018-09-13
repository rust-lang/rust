// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn test_and() {
    let a = true;
    let b = false;
    if a and b {
        //~^ ERROR expected `{`, found `and`
        println!("both");
    }
}

fn test_or() {
    let a = true;
    let b = false;
    if a or b {
        //~^ ERROR expected `{`, found `or`
        println!("both");
    }
}

fn test_and_par() {
    let a = true;
    let b = false;
    if (a and b) {
        //~^ ERROR expected one of `!`, `)`, `,`, `.`, `::`, `?`, `{`, or an operator, found `and`
        println!("both");
    }
}

fn test_or_par() {
    let a = true;
    let b = false;
    if (a or b) {
        //~^ ERROR expected one of `!`, `)`, `,`, `.`, `::`, `?`, `{`, or an operator, found `or`
        println!("both");
    }
}

fn main() {
}

// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #51415: match default bindings were failing to
// see the "move out" implied by `&s` below.

fn main() {
    let a = vec![String::from("a")];
    let opt = a.iter().enumerate().find(|(_, &s)| {
        //~^ ERROR cannot move out
        *s == String::from("d")
    }).map(|(i, _)| i);
    println!("{:?}", opt);
}

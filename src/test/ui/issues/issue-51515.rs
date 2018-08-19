// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

fn main() {
    let foo = &16;
    //~^ HELP consider changing this to be a mutable reference
    //~| SUGGESTION &mut 16
    *foo = 32;
    //~^ ERROR cannot assign to `*foo` which is behind a `&` reference
    let bar = foo;
    //~^ HELP consider changing this to be a mutable reference
    //~| SUGGESTION &mut i32
    *bar = 64;
    //~^ ERROR cannot assign to `*bar` which is behind a `&` reference
}

// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue_19452_aux.rs
extern crate issue_19452_aux;

enum Homura {
    Madoka { age: u32 }
}

fn main() {
    let homura = Homura::Madoka;
    //~^ ERROR expected value, found struct variant `Homura::Madoka`

    let homura = issue_19452_aux::Homura::Madoka;
    //~^ ERROR expected value, found struct variant `issue_19452_aux::Homura::Madoka`
}

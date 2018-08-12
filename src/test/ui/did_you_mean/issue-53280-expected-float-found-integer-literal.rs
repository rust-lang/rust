// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let sixteen: f32 = 16;
    //~^ ERROR mismatched types
    //~| HELP use a float literal
    let a_million_and_seventy: f64 = 1_000_070;
    //~^ ERROR mismatched types
    //~| HELP use a float literal
    let negative_nine: f32 = -9;
    //~^ ERROR mismatched types
    //~| HELP use a float literal


    // only base-10 literals get the suggestion

    let sixteen_again: f64 = 0x10;
    //~^ ERROR mismatched types
    let and_once_more: f32 = 0o20;
    //~^ ERROR mismatched types
}

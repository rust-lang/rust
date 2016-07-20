// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::vec::IntoIter;

pub fn get_tok(it: &mut IntoIter<u8>) {
    let mut found_e = false;

    let temp: Vec<u8> = it.take_while(|&x| {
        found_e = true;
        false
    })
        .cloned()
        //~^ ERROR type mismatch resolving
        //~| expected type `u8`
        //~| found type `&_`
        .collect(); //~ ERROR no method named `collect`
}

fn main() {}

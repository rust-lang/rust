// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

use std::cmp::PartialEq;

trait Hahaha: PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + //~ ERROR duplicate supertrait
              PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq +
              PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq +
              PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq +
              PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq +
              PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq +
              PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq +
              PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq + PartialEq +
              PartialEq {}

struct Lol(int);

impl Hahaha for Lol { }

impl PartialEq for Lol {
    fn eq(&self, other: &Lol) -> bool { **self != **other }
    fn ne(&self, other: &Lol) -> bool { **self == **other }
}

fn main() {
    if Lol(2) == Lol(4) {
        println!("2 == 4");
    } else {
        println!("2 != 4");
    }
}

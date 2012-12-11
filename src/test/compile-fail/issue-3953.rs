// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cmp::Eq;

trait Hahaha: Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq //~ ERROR Duplicate supertrait in trait declaration
              Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq
              Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq
              Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq
              Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq
              Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq Eq {}

enum Lol = int;

pub impl Lol: Hahaha { }

impl Lol: Eq {
    pure fn eq(&self, other: &Lol) -> bool { **self != **other }
    pure fn ne(&self, other: &Lol) -> bool { **self == **other }
}

fn main() {
    if Lol(2) == Lol(4) {
        io::println("2 == 4");
    } else {
        io::println("2 != 4");
    }
}
// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::all)]
#![warn(clippy::if_not_else)]

fn bla() -> bool {
    unimplemented!()
}

fn main() {
    if !bla() {
        println!("Bugs");
    } else {
        println!("Bunny");
    }
    if 4 != 5 {
        println!("Bugs");
    } else {
        println!("Bunny");
    }
}

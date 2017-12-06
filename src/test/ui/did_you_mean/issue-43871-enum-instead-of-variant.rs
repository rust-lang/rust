// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Example { Ex(String), NotEx }

fn result_test() {
    let x = Option(1); //~ ERROR expected function, found enum

    if let Option(_) = x { //~ ERROR expected tuple struct/variant, found enum
        println!("It is OK.");
    }

    let y = Example::Ex(String::from("test"));

    if let Example(_) = y { //~ ERROR expected tuple struct/variant, found enum
        println!("It is OK.");
    }
}

fn main() {}

// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Fruit {
    Apple(String, String),
    Pear(u32),
}


fn main() {
    let x = Fruit::Apple(String::new(), String::new());
    match x {
        Fruit::Apple(a) => {}, //~ ERROR E0023
                               //~| NOTE expected 2 fields, found 1
        Fruit::Apple(a, b, c) => {}, //~ ERROR E0023
                                     //~| NOTE expected 2 fields, found 3
        Fruit::Pear(1, 2) => {}, //~ ERROR E0023
                                 //~| NOTE expected 1 field, found 2
    }
}

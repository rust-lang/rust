// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(dead_assignment)]
#![allow(unused_variables)]

enum Animal {
    Dog (String, f64),
    Cat { name: String, weight: f64 }
}

pub fn main() {
    let mut a: Animal = Animal::Dog("Cocoa".to_string(), 37.2);
    a = Animal::Cat{ name: "Spotty".to_string(), weight: 2.7 };
    // permuting the fields should work too
    let _c = Animal::Cat { weight: 3.1, name: "Spreckles".to_string() };
}

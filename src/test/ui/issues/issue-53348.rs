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
    let mut v = vec!["hello", "this", "is", "a", "test"];

    let v2 = Vec::new();

    v.into_iter().map(|s|s.to_owned()).collect::<Vec<_>>();

    let mut a = String::new();
    for i in v {
        a = *i.to_string();
        //~^ ERROR mismatched types
        //~| NOTE expected struct `std::string::String`, found str
        //~| NOTE expected type
        v2.push(a);
    }
}

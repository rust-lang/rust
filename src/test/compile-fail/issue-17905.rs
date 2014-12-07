// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Show)]
struct Pair<T, V> (T, V);

impl Pair<
    &str, //~ ERROR missing lifetime specifier
    int
> {
    fn say(self: &Pair<&str, int>) {
//~^ ERROR mismatched types: expected `Pair<&'static str, int>`, found `Pair<&str, int>`
        println!("{}", self);
    }
}

fn main() {
    let result = &Pair("shane", 1i);
    result.say();
}

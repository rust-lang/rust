// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
// compile-flags: -Wunused

// ensure there are no special warnings about uninhabited types
// when deriving Debug on an empty enum

#[derive(Debug)]
enum Void {} //~ WARN never used

#[derive(Debug)]
enum Foo { //~ WARN never used
    Bar(u8),
    Void(Void),
}

fn main() {}


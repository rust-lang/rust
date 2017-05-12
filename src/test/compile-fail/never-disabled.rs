// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that ! errors when used in illegal positions with feature(never_type) disabled

// gate-test-never_type

trait Foo {
    type Wub;
}

type Ma = (u32, !, i32); //~ ERROR type is experimental
type Meeshka = Vec<!>; //~ ERROR type is experimental
type Mow = &fn(!) -> !; //~ ERROR type is experimental
type Skwoz = &mut !; //~ ERROR type is experimental

impl Foo for Meeshka {
    type Wub = !; //~ ERROR type is experimental
}

fn main() {
}


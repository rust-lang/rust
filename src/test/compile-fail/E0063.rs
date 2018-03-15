// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

struct SingleFoo {
    x: i32
}

struct PluralFoo {
    x: i32,
    y: i32,
    z: i32
}

struct TruncatedFoo {
    a: i32,
    b: i32,
    x: i32,
    y: i32,
    z: i32
}

struct TruncatedPluralFoo {
    a: i32,
    b: i32,
    c: i32,
    x: i32,
    y: i32,
    z: i32
}


fn main() {
    let w = SingleFoo { };
    //~^ ERROR missing field `x` in initializer of `SingleFoo`
    //~| NOTE missing `x`
    let x = PluralFoo {x: 1};
    //~^ ERROR missing fields `y`, `z` in initializer of `PluralFoo`
    //~| NOTE missing `y`, `z`
    let y = TruncatedFoo{x:1};
    //~^ missing fields `a`, `b`, `y` and 1 other field in initializer of `TruncatedFoo`
    //~| NOTE `a`, `b`, `y` and 1 other field
    let z = TruncatedPluralFoo{x:1};
    //~^ ERROR missing fields `a`, `b`, `c` and 2 other fields in initializer of `TruncatedPluralFoo`
    //~| NOTE missing `a`, `b`, `c` and 2 other fields
}

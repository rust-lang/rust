// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: binary operation + cannot be applied to type
type clam = {x: @int, y: @int};

type fish = {a: @int};

fn main() {
    let a: clam = {x: @1, y: @2};
    let b: clam = {x: @10, y: @20};
    let z: int = a.x + b.y;
    log(debug, z);
    assert (z == 21);
    let forty: fish = {a: @40};
    let two: fish = {a: @2};
    let answer: int = forty.a + two.a;
    log(debug, answer);
    assert (answer == 42);
}

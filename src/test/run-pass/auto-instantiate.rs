// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.





struct Pair<T, U> { a: T, b: U }
struct Triple { x: int, y: int, z: int }

fn f<T,U>(x: T, y: U) -> Pair<T, U> { return Pair {a: x, b: y}; }

pub fn main() {
    println!("{:?}", f(Triple {x: 3, y: 4, z: 5}, 4).a.x);
    println!("{:?}", f(5, 6).a);
}

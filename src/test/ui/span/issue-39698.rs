// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum T {
    T1(i32, i32),
    T2(i32, i32),
    T3(i32),
    T4(i32),
}

fn main() {
    match T::T1(123, 456) {
        T::T1(a, d) | T::T2(d, b) | T::T3(c) | T::T4(a) => { println!("{:?}", a); }
        //~^ ERROR is not bound in all patterns
        //~| ERROR is not bound in all patterns
        //~| ERROR is not bound in all patterns
        //~| ERROR is not bound in all patterns
    }
}

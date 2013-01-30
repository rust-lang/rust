// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn two_args<T>(x: T, y: T) { }

fn main() {
    let mut x: ~[mut int] = ~[3];
    let y: ~[int] = ~[3];
    let a: @mut int = @mut 3;
    let b: @int = @3;

    // NOTE:
    //
    // The fact that this test fails to compile reflects a known
    // shortcoming of the current inference algorithm.  These errors
    // are *not* desirable.

    two_args(x, y); //~ ERROR (values differ in mutability)
    two_args(a, b); //~ ERROR (values differ in mutability)
}
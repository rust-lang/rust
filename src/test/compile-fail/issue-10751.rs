// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that builtin bounds are checked when searching for an
// implementation of a trait
use std::cell::Cell;

trait A {}
trait B {}
trait C {}
trait D {}
impl<T: 'static> A for T {}
impl<T: Send> B for T {}
impl<T: Copy> C for T {}
impl<T: Share> D for T {}

fn main() {
    let a = 3;
    let b = &a;
    let c = &b as &A; //~ ERROR instantiating a type parameter with an incompatible type `&int`
    let d: &A = &b; //~ ERROR instantiating a type parameter with an incompatible type `&int`
    let e = &b as &B; //~ ERROR instantiating a type parameter with an incompatible type `&int`
    let f: &B = &b; //~ ERROR instantiating a type parameter with an incompatible type `&int`
    let g = &~b as &C; //~ ERROR instantiating a type parameter with an incompatible type `~&int`
    let h: &C = &~b; //~ ERROR instantiating a type parameter with an incompatible type `~&int`
    let i = &Cell::new(b) as &D;
    //~^ ERROR instantiating a type parameter with an incompatible type `std::cell::Cell<&int>`
    let j: &D = &Cell::new(b);
    //~^ ERROR instantiating a type parameter with an incompatible type `std::cell::Cell<&int>`

    // These are all ok: int is 'static + Send + Copy + Share
    let k: &A = b;
    let l: &B = b;
    let m: &C = b;
    let n: &D = b;
    let o = b as &A;
    let p = b as &B;
    let q = b as &C;
    let r = b as &D;
}

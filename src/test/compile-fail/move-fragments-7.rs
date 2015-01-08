// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we correctly compute the move fragments for a fn.
//
// Note that the code below is not actually incorrect; the
// `rustc_move_fragments` attribute is a hack that uses the error
// reporting mechanisms as a channel for communicating from the
// internals of the compiler.

// Test that moving a Box<T> fragments its containing structure, for
// both moving out of the structure (i.e. reading `*p.x`) and writing
// into the container (i.e. writing `*p.x`).

pub struct D { d: isize }
impl Drop for D { fn drop(&mut self) { } }

pub struct Pair<X,Y> { x: X, y: Y }

#[rustc_move_fragments]
pub fn test_deref_box_field(p: Pair<Box<D>, Box<D>>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                 parent_of_fragments: `$(local p).x`
    //~| ERROR                     moved_leaf_path: `$(local p).x.*`
    //~| ERROR                    unmoved_fragment: `$(local p).y`
    //~| ERROR                  assigned_leaf_path: `$(local i)`
    let i : D = *p.x;
}

#[rustc_move_fragments]
pub fn test_overwrite_deref_box_field(mut p: Pair<Box<D>, Box<D>>) {
    //~^ ERROR                 parent_of_fragments: `$(local mut p)`
    //~| ERROR                 parent_of_fragments: `$(local mut p).x`
    //~| ERROR                  assigned_leaf_path: `$(local mut p).x.*`
    //~| ERROR                    unmoved_fragment: `$(local mut p).y`
    *p.x = D { d: 3 };
}

pub fn main() { }

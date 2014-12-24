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

// Test that moving into a field (i.e. overwriting it) fragments the
// receiver.

use std::mem::drop;

pub struct Pair<X,Y> { x: X, y: Y }

#[rustc_move_fragments]
pub fn test_overwrite_uninit_field<Z>(z: Z) {
    //~^ ERROR                 parent_of_fragments: `$(local mut p)`
    //~| ERROR                  assigned_leaf_path: `$(local z)`
    //~| ERROR                     moved_leaf_path: `$(local z)`
    //~| ERROR                  assigned_leaf_path: `$(local mut p).x`
    //~| ERROR                    unmoved_fragment: `$(local mut p).y`

    let mut p: Pair<Z,Z>;
    p.x = z;
}

#[rustc_move_fragments]
pub fn test_overwrite_moved_field<Z>(mut p: Pair<Z,Z>, z: Z) {
    //~^ ERROR                 parent_of_fragments: `$(local mut p)`
    //~| ERROR                  assigned_leaf_path: `$(local z)`
    //~| ERROR                     moved_leaf_path: `$(local z)`
    //~| ERROR                  assigned_leaf_path: `$(local mut p).y`
    //~| ERROR                    unmoved_fragment: `$(local mut p).x`

    drop(p);
    p.y = z;
}

#[rustc_move_fragments]
pub fn test_overwrite_same_field<Z>(mut p: Pair<Z,Z>) {
    //~^ ERROR                 parent_of_fragments: `$(local mut p)`
    //~| ERROR                     moved_leaf_path: `$(local mut p).x`
    //~| ERROR                  assigned_leaf_path: `$(local mut p).x`
    //~| ERROR                    unmoved_fragment: `$(local mut p).y`

    p.x = p.x;
}

pub fn main() { }

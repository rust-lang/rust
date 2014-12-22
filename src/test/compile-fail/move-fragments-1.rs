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

// These are all fairly trivial cases: unused variables or direct
// drops of substructure.

pub struct D { d: int }
impl Drop for D { fn drop(&mut self) { } }

#[rustc_move_fragments]
pub fn test_noop() {
}

#[rustc_move_fragments]
pub fn test_take(_x: D) {
    //~^ ERROR                  assigned_leaf_path: `$(local _x)`
}

pub struct Pair<X,Y> { x: X, y: Y }

#[rustc_move_fragments]
pub fn test_take_struct(_p: Pair<D, D>) {
    //~^ ERROR                  assigned_leaf_path: `$(local _p)`
}

#[rustc_move_fragments]
pub fn test_drop_struct_part(p: Pair<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                     moved_leaf_path: `$(local p).x`
    //~| ERROR                    unmoved_fragment: `$(local p).y`
    drop(p.x);
}

#[rustc_move_fragments]
pub fn test_drop_tuple_part(p: (D, D)) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                     moved_leaf_path: `$(local p).#0`
    //~| ERROR                    unmoved_fragment: `$(local p).#1`
    drop(p.0);
}

pub fn main() { }

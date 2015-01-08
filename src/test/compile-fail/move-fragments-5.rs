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

// This is the first test that checks moving into local variables.

pub struct D { d: isize }
impl Drop for D { fn drop(&mut self) { } }

pub struct Pair<X,Y> { x: X, y: Y }

#[rustc_move_fragments]
pub fn test_move_field_to_local(p: Pair<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                     moved_leaf_path: `$(local p).x`
    //~| ERROR                    unmoved_fragment: `$(local p).y`
    //~| ERROR                  assigned_leaf_path: `$(local _x)`
    let _x = p.x;
}

#[rustc_move_fragments]
pub fn test_move_field_to_local_to_local(p: Pair<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                     moved_leaf_path: `$(local p).x`
    //~| ERROR                    unmoved_fragment: `$(local p).y`
    //~| ERROR                  assigned_leaf_path: `$(local _x)`
    //~| ERROR                     moved_leaf_path: `$(local _x)`
    //~| ERROR                  assigned_leaf_path: `$(local _y)`
    let _x = p.x;
    let _y = _x;
}

// In the following fn's `test_move_field_to_local_delayed` and
// `test_uninitialized_local` , the instrumentation reports that `_x`
// is moved. This is unlike `test_move_field_to_local`, where `_x` is
// just reported as an assigned_leaf_path. Presumably because this is
// how we represent that it did not have an initializing expression at
// the binding site.

#[rustc_move_fragments]
pub fn test_uninitialized_local(_p: Pair<D, D>) {
    //~^ ERROR                  assigned_leaf_path: `$(local _p)`
    //~| ERROR                     moved_leaf_path: `$(local _x)`
    let _x: D;
}

#[rustc_move_fragments]
pub fn test_move_field_to_local_delayed(p: Pair<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                     moved_leaf_path: `$(local p).x`
    //~| ERROR                    unmoved_fragment: `$(local p).y`
    //~| ERROR                  assigned_leaf_path: `$(local _x)`
    //~| ERROR                     moved_leaf_path: `$(local _x)`
    let _x;
    _x = p.x;
}

#[rustc_move_fragments]
pub fn test_move_field_mut_to_local(mut p: Pair<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local mut p)`
    //~| ERROR                     moved_leaf_path: `$(local mut p).x`
    //~| ERROR                    unmoved_fragment: `$(local mut p).y`
    //~| ERROR                  assigned_leaf_path: `$(local _x)`
    let _x = p.x;
}

#[rustc_move_fragments]
pub fn test_move_field_to_local_to_local_mut(p: Pair<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                     moved_leaf_path: `$(local p).x`
    //~| ERROR                    unmoved_fragment: `$(local p).y`
    //~| ERROR                  assigned_leaf_path: `$(local mut _x)`
    //~| ERROR                     moved_leaf_path: `$(local mut _x)`
    //~| ERROR                  assigned_leaf_path: `$(local _y)`
    let mut _x = p.x;
    let _y = _x;
}

pub fn main() {}

// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
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

// These are checking that enums are tracked; note that their output
// paths include "downcasts" of the path to a particular enum.

#![feature(rustc_attrs)]

use self::Lonely::{Zero, One, Two};

pub struct D { d: isize }
impl Drop for D { fn drop(&mut self) { } }

pub enum Lonely<X,Y> { Zero, One(X), Two(X, Y) }

#[rustc_move_fragments]
pub fn test_match_partial(p: Lonely<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                  assigned_leaf_path: `($(local p) as Lonely::Zero)`
    match p {
        Zero => {}
        _ => {}
    }
}

#[rustc_move_fragments]
pub fn test_match_full(p: Lonely<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                  assigned_leaf_path: `($(local p) as Lonely::Zero)`
    //~| ERROR                  assigned_leaf_path: `($(local p) as Lonely::One)`
    //~| ERROR                  assigned_leaf_path: `($(local p) as Lonely::Two)`
    match p {
        Zero => {}
        One(..) => {}
        Two(..) => {}
    }
}

#[rustc_move_fragments]
pub fn test_match_bind_one(p: Lonely<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                  assigned_leaf_path: `($(local p) as Lonely::Zero)`
    //~| ERROR                 parent_of_fragments: `($(local p) as Lonely::One)`
    //~| ERROR                     moved_leaf_path: `($(local p) as Lonely::One).#0`
    //~| ERROR                  assigned_leaf_path: `($(local p) as Lonely::Two)`
    //~| ERROR                  assigned_leaf_path: `$(local data)`
    match p {
        Zero => {}
        One(data) => {}
        Two(..) => {}
    }
}

#[rustc_move_fragments]
pub fn test_match_bind_many(p: Lonely<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                  assigned_leaf_path: `($(local p) as Lonely::Zero)`
    //~| ERROR                 parent_of_fragments: `($(local p) as Lonely::One)`
    //~| ERROR                     moved_leaf_path: `($(local p) as Lonely::One).#0`
    //~| ERROR                  assigned_leaf_path: `$(local data)`
    //~| ERROR                 parent_of_fragments: `($(local p) as Lonely::Two)`
    //~| ERROR                     moved_leaf_path: `($(local p) as Lonely::Two).#0`
    //~| ERROR                     moved_leaf_path: `($(local p) as Lonely::Two).#1`
    //~| ERROR                  assigned_leaf_path: `$(local left)`
    //~| ERROR                  assigned_leaf_path: `$(local right)`
    match p {
        Zero => {}
        One(data) => {}
        Two(left, right) => {}
    }
}

pub fn main() { }

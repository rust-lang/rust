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

// This checks the handling of `_` within variants, especially when mixed
// with bindings.

#![feature(rustc_attrs)]

use self::Lonely::{Zero, One, Two};

pub struct D { d: isize }
impl Drop for D { fn drop(&mut self) { } }

pub enum Lonely<X,Y> { Zero, One(X), Two(X, Y) }

#[rustc_move_fragments]
pub fn test_match_bind_and_underscore(p: Lonely<D, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local p)`
    //~| ERROR                  assigned_leaf_path: `($(local p) as Lonely::Zero)`
    //~| ERROR                  assigned_leaf_path: `($(local p) as Lonely::One)`
    //~| ERROR                 parent_of_fragments: `($(local p) as Lonely::Two)`
    //~| ERROR                     moved_leaf_path: `($(local p) as Lonely::Two).#0`
    //~| ERROR                    unmoved_fragment: `($(local p) as Lonely::Two).#1`
    //~| ERROR                  assigned_leaf_path: `$(local left)`

    match p {
        Zero => {}

        One(_) => {}       // <-- does not fragment `($(local p) as One)` ...

        Two(left, _) => {} // <-- ... *does* fragment `($(local p) as Two)`.
    }
}

pub fn main() { }

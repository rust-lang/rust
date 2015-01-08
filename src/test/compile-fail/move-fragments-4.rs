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

// This checks that a move of deep structure is properly tracked. (An
// early draft of the code did not properly traverse up through all of
// the parents of the leaf fragment.)

pub struct D { d: isize }
impl Drop for D { fn drop(&mut self) { } }

pub struct Pair<X,Y> { x: X, y: Y }

#[rustc_move_fragments]
pub fn test_move_substructure(pppp: Pair<Pair<Pair<Pair<D,D>, D>, D>, D>) {
    //~^ ERROR                 parent_of_fragments: `$(local pppp)`
    //~| ERROR                 parent_of_fragments: `$(local pppp).x`
    //~| ERROR                 parent_of_fragments: `$(local pppp).x.x`
    //~| ERROR                    unmoved_fragment: `$(local pppp).x.x.x`
    //~| ERROR                     moved_leaf_path: `$(local pppp).x.x.y`
    //~| ERROR                    unmoved_fragment: `$(local pppp).x.y`
    //~| ERROR                    unmoved_fragment: `$(local pppp).y`
    drop(pppp.x.x.y);
}

pub fn main() { }

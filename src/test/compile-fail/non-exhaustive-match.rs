// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]

enum t { a, b, }

fn main() {
    let x = t::a;
    match x { t::b => { } } //~ ERROR non-exhaustive patterns: `a` not covered
    match true { //~ ERROR non-exhaustive patterns: `false` not covered
      true => {}
    }
    match Some(10) { //~ ERROR non-exhaustive patterns: `Some(_)` not covered
      None => {}
    }
    match (2, 3, 4) { //~ ERROR non-exhaustive patterns: `(_, _, _)` not covered
      (_, _, 4) => {}
    }
    match (t::a, t::a) { //~ ERROR non-exhaustive patterns: `(a, a)` not covered
      (t::a, t::b) => {}
      (t::b, t::a) => {}
    }
    match t::a { //~ ERROR non-exhaustive patterns: `b` not covered
      t::a => {}
    }
    // This is exhaustive, though the algorithm got it wrong at one point
    match (t::a, t::b) {
      (t::a, _) => {}
      (_, t::a) => {}
      (t::b, t::b) => {}
    }
    let vec = vec![Some(42), None, Some(21)];
    let vec: &[Option<isize>] = &vec;
    match *vec { //~ ERROR non-exhaustive patterns: `[]` not covered
        [Some(..), None, ref tail..] => {}
        [Some(..), Some(..), ref tail..] => {}
        [None] => {}
    }
    let vec = vec![1];
    let vec: &[isize] = &vec;
    match *vec {
        [_, ref tail..] => (),
        [] => ()
    }
    let vec = vec![0.5f32];
    let vec: &[f32] = &vec;
    match *vec { //~ ERROR non-exhaustive patterns: `[_, _, _, _]` not covered
        [0.1, 0.2, 0.3] => (),
        [0.1, 0.2] => (),
        [0.1] => (),
        [] => ()
    }
    let vec = vec![Some(42), None, Some(21)];
    let vec: &[Option<isize>] = &vec;
    match *vec {
        [Some(..), None, ref tail..] => {}
        [Some(..), Some(..), ref tail..] => {}
        [None, None, ref tail..] => {}
        [None, Some(..), ref tail..] => {}
        [Some(_)] => {}
        [None] => {}
        [] => {}
    }
}

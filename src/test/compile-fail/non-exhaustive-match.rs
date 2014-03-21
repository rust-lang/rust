// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum t { a, b, }

fn main() {
    let x = a;
    match x { b => { } } //~ ERROR non-exhaustive patterns
    match true { //~ ERROR non-exhaustive patterns
      true => {}
    }
    match Some(10) { //~ ERROR non-exhaustive patterns
      None => {}
    }
    match (2, 3, 4) { //~ ERROR non-exhaustive patterns
      (_, _, 4) => {}
    }
    match (a, a) { //~ ERROR non-exhaustive patterns
      (a, b) => {}
      (b, a) => {}
    }
    match a { //~ ERROR b not covered
      a => {}
    }
    // This is exhaustive, though the algorithm got it wrong at one point
    match (a, b) {
      (a, _) => {}
      (_, a) => {}
      (b, b) => {}
    }
    let vec = vec!(Some(42), None, Some(21));
    let vec: &[Option<int>] = vec.as_slice();
    match vec {
        //~^ ERROR non-exhaustive patterns: vectors of length 0 not covered
        [Some(..), None, ..tail] => {}
        [Some(..), Some(..), ..tail] => {}
        [None] => {}
    }
    let vec = vec!(1);
    let vec: &[int] = vec.as_slice();
    match vec {
        [_, ..tail] => (),
        [] => ()
    }
    let vec = vec!(0.5);
    let vec: &[f32] = vec.as_slice();
    match vec { //~ ERROR non-exhaustive patterns: vectors of length 4 not covered
        [0.1, 0.2, 0.3] => (),
        [0.1, 0.2] => (),
        [0.1] => (),
        [] => ()
    }
    let vec = vec!(Some(42), None, Some(21));
    let vec: &[Option<int>] = vec.as_slice();
    match vec {
        [Some(..), None, ..tail] => {}
        [Some(..), Some(..), ..tail] => {}
        [None, None, ..tail] => {}
        [None, Some(..), ..tail] => {}
        [Some(_)] => {}
        [None] => {}
        [] => {}
    }
}

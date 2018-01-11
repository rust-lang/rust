// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Enum {
    A, B, C, D, E, F
}
use Enum::*;

struct S(Enum, ());
struct Sd { x: Enum, y: () }

fn main() {
    match (A, ()) { //~ ERROR non-exhaustive
        (A, _) => {}
    }

    match (A, A) { //~ ERROR non-exhaustive
        (_, A) => {}
    }

    match ((A, ()), ()) { //~ ERROR non-exhaustive
        ((A, ()), _) => {}
    }

    match ((A, ()), A) { //~ ERROR non-exhaustive
        ((A, ()), _) => {}
    }

    match ((A, ()), ()) { //~ ERROR non-exhaustive
        ((A, _), _) => {}
    }


    match S(A, ()) { //~ ERROR non-exhaustive
        S(A, _) => {}
    }

    match (Sd { x: A, y: () }) { //~ ERROR non-exhaustive
        Sd { x: A, y: _ } => {}
    }

    match Some(A) { //~ ERROR non-exhaustive
        Some(A) => (),
        None => ()
    }
}

// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]
#![deny(unreachable_patterns)]

// The arity of `ref x` is always 1. If the pattern is compared to some non-structural type whose
// arity is always 0, an ICE occurs.
//
// Related issue: #23009

fn main() {
    let homura = [1, 2, 3];

    match homura {
        [1, ref _madoka, 3] => (),
        [1, 2, 3] => (), //~ ERROR unreachable pattern
        [_, _, _] => (),
    }
}

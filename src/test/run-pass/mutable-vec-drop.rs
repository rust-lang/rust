// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct Pair { a: int, b: int}

fn main() {
    // This just tests whether the vec leaks its members.
    let mut pvec: ~[@Pair] =
        ~[@Pair{a: 1, b: 2}, @Pair{a: 3, b: 4}, @Pair{a: 5, b: 6}];
}

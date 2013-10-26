// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.





struct Large {a: int,
             b: int,
             c: int,
             d: int,
             e: int,
             f: int,
             g: int,
             h: int,
             i: int,
             j: int,
             k: int,
             l: int}
fn f() {
    let _foo: Large =
        Large {a: 0,
         b: 0,
         c: 0,
         d: 0,
         e: 0,
         f: 0,
         g: 0,
         h: 0,
         i: 0,
         j: 0,
         k: 0,
         l: 0};
}

pub fn main() { f(); }

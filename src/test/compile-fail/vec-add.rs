// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test

// FIXME (Issue #2711): + should allow immutable or mutable vectors on
// the right hand side in all cases. We are getting compiler errors
// about this now, so I'm ignoring the test for now. -eholk

fn add(i: ~[int], mut m: ~[int]) {

    // Check that:
    //  (1) vectors of any two mutabilities can be added
    //  (2) result has mutability of lhs

   add(i + ~[3],
       m + ~[3],
       ~[3]);

   add(i + ~[3],
       m + ~[3],
       ~[3]);

   add(i + i,
       m + i,
       i);

   add(i + m,
       m + m,
       m);

   add(m + ~[3], //~ ERROR mismatched types
       m + ~[3],
       m + ~[3]);

   add(i + ~[3],
       i + ~[3], //~ ERROR mismatched types
       i + ~[3]);

   add(m + ~[3], //~ ERROR mismatched types
       m + ~[3],
       m + ~[3]);

   add(i + ~[3],
       i + ~[3], //~ ERROR mismatched types
       i + ~[3]);

   add(m + i, //~ ERROR mismatched types
       m + i,
       m + i);

   add(i + i,
       i + i, //~ ERROR mismatched types
       i + i);

   add(m + m, //~ ERROR mismatched types
       m + m,
       m + m);

   add(i + m,
       i + m, //~ ERROR mismatched types
       i + m);
}

fn main() {
}

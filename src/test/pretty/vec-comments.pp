// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #679
// Testing that comments are correctly interleaved
// pp-exact:vec-comments.pp
fn main() {
    let _v1 =
        ~[
          // Comment
          0,
          // Comment
          1,
          // Comment
          2];
    let _v2 =
        ~[0, // Comment
          1, // Comment
          2]; // Comment
    let _v3 =
        ~[
          /* Comment */
          0,
          /* Comment */
          1,
          /* Comment */
          2];
    let _v4 =
        ~[0, /* Comment */
          1, /* Comment */
          2]; /* Comment */
}

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:

/* This is a test to ensure that we do _not_ support nested/balanced comments. I know you might be
   thinking "but nested comments are cool", and that would be a valid point, but they are also a
   thing that would make our lexical syntax non-regular, and we do not want that to be true.

   omitting-things at a higher level (tokens) should be done via token-trees / macros,
   not comments.

   /*
     fail here
   */
*/

fn main() {
}

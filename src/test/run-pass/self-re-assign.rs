// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure assigning an owned or managed variable to itself works. In particular,
// that we do not glue_drop before we glue_take (#3290).

use std::rc::Rc;

pub fn main() {
   let mut x = ~3;
   x = x;
   assert!(*x == 3);

   let mut x = Rc::new(3);
   x = x;
   assert!(*x.borrow() == 3);
}

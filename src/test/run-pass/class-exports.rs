// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* Test that exporting a class also exports its
   public fields and methods */

use kitty::*;

mod kitty {
    pub struct cat {
        meows: uint,
        name: ~str,
    }

    pub impl cat {
        fn get_name(&self) -> ~str { self.name.clone() }
    }

    pub fn cat(in_name: ~str) -> cat {
        cat {
            name: in_name,
            meows: 0u
        }
    }
}

pub fn main() {
  assert_eq!(cat(~"Spreckles").get_name(), ~"Spreckles");
}

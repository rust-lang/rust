// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//error-pattern:libc::c_int or libc::c_long should be used
mod xx {
    extern {
        pub fn strlen(str: *u8) -> uint;
        pub fn foo(x: int, y: uint);
    }
}

fn main() {
  // let it fail to verify warning message
  fail!()
}

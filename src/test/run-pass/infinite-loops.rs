// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
  A simple way to make sure threading works. This should use all the
  CPU cycles an any machines that we're likely to see for a while.
*/
// xfail-test

extern mod std;

fn loopy(n: int) {
    if n > 0 { do spawn { loopy(n - 1) }; do spawn { loopy(n - 1) }; }
    loop { }
}

pub fn main() { 
    // Commenting this out, as this will hang forever otherwise.
    // Even after seeing the comment above, I'm not sure what the
    // intention of this test is.
    // do spawn { loopy(5) }; 
}

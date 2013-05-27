// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test will call __morestack with various minimum stack sizes

extern mod extra;

fn getbig(i: int) {
    if i != 0 {
        getbig(i - 1);
    }
}

pub fn main() {
    let mut sz = 400u;
    while sz < 500u {
        task::try(|| getbig(200) );
        sz += 1u;
    }
}

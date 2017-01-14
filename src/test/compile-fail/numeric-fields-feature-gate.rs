// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-relaxed_adts

struct S(u8);

fn main() {
    let s = S{0: 10}; //~ ERROR numeric fields in struct expressions are unstable
    match s {
        S{0: a, ..} => {} //~ ERROR numeric fields in struct patterns are unstable
    }
}

// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-rustfix
// compile-flags: -Z continue-parse-after-error

extern "C" {
    const C: u8; //~ ERROR extern items cannot be `const`
}

fn main() {
    // We suggest turning the (illegal) extern `const` into an extern `static`,
    // but this also requires `unsafe` (a deny-by-default lint at comment time,
    // future error; Issue #36247)
    unsafe {
        let _x = C;
    }
}

// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern "C" {
    fn printf(_: *const u8, ...) -> u32;
    //~^ NOTE defined here
}

fn main() {
    unsafe { printf(); }
    //~^ ERROR E0060
    //~| expected at least 1 parameter
}

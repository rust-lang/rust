// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Delicious {
    Pie      = 0x1,
    Apple    = 0x2,
    ApplePie = Delicious::Apple as isize | Delicious::PIE as isize,
    //~^ ERROR no associated item named `PIE` found for type `Delicious`
}

const FOO: [u32; u8::MIN as usize] = [];
//~^ ERROR no associated item named `MIN` found for type `u8`

fn main() {}

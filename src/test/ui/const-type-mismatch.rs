// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `const`s shouldn't suggest `.into()`

const TEN: u8 = 10;
const TWELVE: u16 = TEN + 2;
//~^ ERROR mismatched types [E0308]

fn main() {
    const TEN: u8 = 10;
    const ALSO_TEN: u16 = TEN;
    //~^ ERROR mismatched types [E0308]
}

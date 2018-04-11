// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    // Tests case where inference fails due to the order in which casts are checked.
    // Ideally this would compile, see #48270.
    let x = &"hello";
    let mut y = 0 as *const _;
    //~^ ERROR cannot cast to a pointer of an unknown kind
    y = x as *const _;
}

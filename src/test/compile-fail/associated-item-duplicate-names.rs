// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for issue #23969


trait Foo {
    type Ty;
    const BAR: u32;
}

impl Foo for () {
    type Ty = ();
    type Ty = usize; //~ ERROR duplicate definitions
    const BAR: u32 = 7;
    const BAR: u32 = 8; //~ ERROR duplicate definitions
}

fn main() {
    let _: <() as Foo>::Ty = ();
    let _: u32 = <() as Foo>::BAR;
}

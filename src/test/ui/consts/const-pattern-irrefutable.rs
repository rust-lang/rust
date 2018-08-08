// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub const b: u8 = 2;
    pub const d: u8 = 2;
}

use foo::b as c;
use foo::d;

const a: u8 = 2;

fn main() {
    let a = 4; //~ ERROR refutable pattern in local binding: `_` not covered
    let c = 4; //~ ERROR refutable pattern in local binding: `_` not covered
    let d = 4; //~ ERROR refutable pattern in local binding: `_` not covered
    fn f() {} // Check that the `NOTE`s still work with an item here (c.f. issue #35115).
}

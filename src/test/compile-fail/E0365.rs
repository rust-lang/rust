// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub const X: u32 = 1;
}

pub use foo as foo2;
//~^ ERROR `foo` is private, and cannot be reexported [E0365]
//~| NOTE reexport of private `foo`
//~| NOTE consider declaring type or module `foo` with `pub`

fn main() {}

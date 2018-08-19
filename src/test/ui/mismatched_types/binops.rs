// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    1 + Some(1); //~ ERROR cannot add `std::option::Option<{integer}>` to `{integer}`
    2 as usize - Some(1); //~ ERROR cannot subtract `std::option::Option<{integer}>` from `usize`
    3 * (); //~ ERROR cannot multiply `()` to `{integer}`
    4 / ""; //~ ERROR cannot divide `{integer}` by `&str`
    5 < String::new(); //~ ERROR can't compare `{integer}` with `std::string::String`
    6 == Ok(1); //~ ERROR can't compare `{integer}` with `std::result::Result<{integer}, _>`
}

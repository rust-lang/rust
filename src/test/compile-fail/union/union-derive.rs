// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Most traits cannot be derived for unions.

#![feature(untagged_unions)]

#[derive(
    PartialEq, //~ ERROR this trait cannot be derived for unions
    PartialOrd, //~ ERROR this trait cannot be derived for unions
    Ord, //~ ERROR this trait cannot be derived for unions
    Hash, //~ ERROR this trait cannot be derived for unions
    Default, //~ ERROR this trait cannot be derived for unions
    Debug, //~ ERROR this trait cannot be derived for unions
)]
union U {
    a: u8,
    b: u16,
}

fn main() {}

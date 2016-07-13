// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

type foo = fn(&u8, &u8) -> &u8; //~ ERROR missing lifetime specifier
//~^ HELP the signature does not say whether it is borrowed from argument 1 or argument 2

fn bar<F: Fn(&u8, &u8) -> &u8>(f: &F) {} //~ ERROR missing lifetime specifier
//~^ HELP the signature does not say whether it is borrowed from argument 1 or argument 2

fn main() {}

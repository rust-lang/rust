// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The raw_pointer_derived lint warns about its removal
// cc #30346

// compile-flags:-D raw_pointer_derive

// error-pattern:lint `raw_pointer_derive` has been removed
// error-pattern:requested on the command line with `-D raw_pointer_derive`

#![warn(unused)]

#[deny(warnings)]
fn main() { let unused = (); }

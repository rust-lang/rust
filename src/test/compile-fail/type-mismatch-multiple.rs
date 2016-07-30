// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checking that the compiler reports multiple type errors at once

fn main() { let a: bool = 1; let b: i32 = true; }
//~^ ERROR mismatched types
//~| expected type `bool`
//~| found type `{integer}`
//~| expected bool, found integral variable
//~| ERROR mismatched types
//~| expected i32, found bool


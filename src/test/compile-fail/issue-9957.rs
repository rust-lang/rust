// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub extern crate std; //~ ERROR: `pub` visibility is not allowed
priv extern crate std; //~ ERROR: unnecessary visibility qualifier
extern crate std;

pub use std::bool;
priv use std::bool; //~ ERROR: unnecessary visibility qualifier
use std::bool;

fn main() {
    pub use std::bool; //~ ERROR: imports in functions are never reachable
    priv use std::bool; //~ ERROR: unnecessary visibility qualifier
    use std::bool;
}

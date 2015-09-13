// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Unstable entities should be caught in import lists

#![allow(unused_imports)]

use std::thread::{catch_panic, ScopedKey}; //~ ERROR use of unstable library feature 'catch_panic'
//~^ ERROR use of unstable library feature 'scoped_tls'

use std::rt::{self}; //~ ERROR use of unstable library feature 'rt'

fn main() {
}

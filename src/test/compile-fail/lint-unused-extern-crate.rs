// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs)]
#![deny(unused_extern_crate)]
#![allow(unused_variable)]

extern crate libc; //~ ERROR: unused extern crate

extern crate "collections" as collecs; // no error, it is used

extern crate rand; // no error, the use marks it as used
                   // even if imported objects aren't used

extern crate time; // no error, the use * marks it as used

#[allow(unused_imports)]
use rand::isaac::IsaacRng;

use time::*;

fn main() {
    let x: collecs::vec::Vec<uint> = Vec::new();
    let y = now();
}

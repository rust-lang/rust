// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

use std::ops::{Add, Sub};

type Test = Add +
            //~^ ERROR E0393
            //~| NOTE missing reference to `RHS`
            //~| NOTE because of the default `Self` reference, type parameters must be specified on object types
            //~| ERROR E0191
            //~| NOTE missing associated type `Output` value
            Sub;
            //~^ ERROR E0225
            //~| NOTE non-Send/Sync additional trait

fn main() { }

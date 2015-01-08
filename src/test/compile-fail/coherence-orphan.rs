// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:coherence-orphan-lib.rs

extern crate "coherence-orphan-lib" as lib;

use lib::TheTrait;

struct TheType;

impl TheTrait<usize> for isize { } //~ ERROR E0117

impl TheTrait<TheType> for isize { } //~ ERROR E0117

impl TheTrait<isize> for TheType { }

fn main() { }

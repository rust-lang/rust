// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Each path node in a `use` declaration must be treated as an item. If not, the following code
// will trigger an ICE.
//
// Related issue: #25763

use std::{mem, ptr};
use std::mem; //~ ERROR the name `mem` is defined multiple times

fn main() {}

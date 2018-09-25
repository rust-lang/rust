// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::{Debug, Display};

#[marker] trait ExplicitMarker {}
//~^ ERROR marker traits is an experimental feature (see issue #29864)

impl<T: Display> ExplicitMarker for T {}
impl<T: Debug> ExplicitMarker for T {}

fn main() {}

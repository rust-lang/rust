// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:complex_impl_support.rs

extern crate complex_impl_support;

use complex_impl_support::{External, M};

struct Q;

impl<R> External for (Q, R) {} //~ ERROR must be used
//~^ ERROR conflicting implementations of trait

fn main() {}

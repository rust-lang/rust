// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Debug;

trait MultiRegionTrait<'a, 'b> {}
impl<'a, 'b> MultiRegionTrait<'a, 'b> for (&'a u32, &'b u32) {}

fn no_least_region<'a, 'b>(x: &'a u32, y: &'b u32) -> impl MultiRegionTrait<'a, 'b> {
//~^ ERROR ambiguous lifetime bound
    (x, y)
}

fn main() {}

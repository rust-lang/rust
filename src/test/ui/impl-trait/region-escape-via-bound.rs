// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do not allow the region `'x` to escape in the impl
// trait **even though** `'y` escapes, which outlives `'x`.
//
// See https://github.com/rust-lang/rust/issues/46541 for more details.

#![allow(dead_code)]
#![feature(in_band_lifetimes)]
#![feature(nll)]

use std::cell::Cell;

trait Trait<'a> { }

impl Trait<'b> for Cell<&'a u32> { }

fn foo(x: Cell<&'x u32>) -> impl Trait<'y>
    //~^ ERROR hidden type for `impl Trait` captures lifetime that does not appear in bounds [E0700]
where 'x: 'y
{
    x
}

fn main() { }

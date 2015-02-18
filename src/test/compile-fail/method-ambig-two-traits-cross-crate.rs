// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test an ambiguity scenario where one copy of the method is available
// from a trait imported from another crate.

// aux-build:ambig_impl_2_lib.rs
extern crate ambig_impl_2_lib;
use ambig_impl_2_lib::me;
trait me2 {
    fn me(&self) -> usize;
}
impl me2 for usize { fn me(&self) -> usize { *self } }
fn main() { 1_usize.me(); } //~ ERROR E0034


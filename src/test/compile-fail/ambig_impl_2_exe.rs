// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast aux-build
// aux-build:ambig_impl_2_lib.rs
extern mod ambig_impl_2_lib;
use ambig_impl_2_lib::me;
trait me {
    fn me(&self) -> uint;
}
impl me for uint { fn me(&self) -> uint { *self } } //~ NOTE is `me$uint::me`
fn main() { 1u.me(); } //~ ERROR multiple applicable methods in scope
//~^ NOTE is `ambig_impl_2_lib::me$uint::me`

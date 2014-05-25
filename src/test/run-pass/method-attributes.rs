// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pp-exact - Make sure we print all the attributes
#![allow(unused_attribute)]

#[frobable]
trait frobable {
    #[frob_attr]
    fn frob(&self);
    #[defrob_attr]
    fn defrob(&self);
}

#[int_frobable]
impl frobable for int {
    #[frob_attr1]
    fn frob(&self) {
        #![frob_attr2]
    }

    #[defrob_attr1]
    fn defrob(&self) {
        #![defrob_attr2]
    }
}

pub fn main() { }

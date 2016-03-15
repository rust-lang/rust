// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that provided items are inherited properly even when impls vary in
// type parameters *and* rely on projections.

#![feature(specialization)]

use std::convert::Into;

trait Trait {
    fn to_u8(&self) -> u8;
}
trait WithAssoc {
    type Item;
    fn to_item(&self) -> Self::Item;
}

impl<T, U> Trait for T where T: WithAssoc<Item=U>, U: Into<u8> {
    fn to_u8(&self) -> u8 {
        self.to_item().into()
    }
}

impl WithAssoc for u8 {
    type Item = u8;
    fn to_item(&self) -> u8 { *self }
}

impl Trait for u8 {}

fn main() {
    assert!(3u8.to_u8() == 3u8);
}

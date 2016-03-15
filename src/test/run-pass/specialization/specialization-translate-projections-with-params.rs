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
// type parameters *and* rely on projections, and the type parameters are input
// types on the trait.

#![feature(specialization)]

trait Trait<T> {
    fn convert(&self) -> T;
}
trait WithAssoc {
    type Item;
    fn as_item(&self) -> &Self::Item;
}

impl<T, U> Trait<U> for T where T: WithAssoc<Item=U>, U: Clone {
    fn convert(&self) -> U {
        self.as_item().clone()
    }
}

impl WithAssoc for u8 {
    type Item = u8;
    fn as_item(&self) -> &u8 { self }
}

impl Trait<u8> for u8 {}

fn main() {
    assert!(3u8.convert() == 3u8);
}

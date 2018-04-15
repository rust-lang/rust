// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Formerly this ICEd with the following message:
// Tried to project an inherited associated type during coherence checking,
// which is currently not supported.
//
// No we expect to run into a more user-friendly cycle error instead.

#![feature(specialization)]

trait Trait<T> { type Assoc; }
//~^ cycle detected

impl<T> Trait<T> for Vec<T> {
    type Assoc = ();
}

impl Trait<u8> for Vec<u8> {}

impl<T> Trait<T> for String {
    type Assoc = ();
}

impl Trait<<Vec<u8> as Trait<u8>>::Assoc> for String {}

fn main() {}

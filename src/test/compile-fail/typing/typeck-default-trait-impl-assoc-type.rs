// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do not consider associated types to be sendable without
// some applicable trait bound (and we don't ICE).

trait Trait {
    type AssocType;
    fn dummy(&self) { }
}
fn bar<T:Trait+Send>() {
    is_send::<T::AssocType>(); //~ ERROR E0277
}

fn is_send<T:Send>() {
}

fn main() { }

// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(index_assign_trait)]

use std::ops::IndexAssign;

struct Array;

impl IndexAssign<(), ()> for Array {
    fn index_assign(&mut self, _: (), _: ()) {
        unimplemented!()
    }
}

fn main() {
    let mut array = Array;
    array[()] = ();
    //~^ error: overloaded indexed assignments are not stable
    //~| help: add `#![feature(indexed_assignments)]` to the crate features to enable
//error: overloaded indexed assignemnts are not stable
}

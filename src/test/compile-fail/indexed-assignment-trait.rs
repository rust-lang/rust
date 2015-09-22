// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

use std::ops::IndexAssign;
//~^ error: use of unstable library feature 'index_assign_trait'

struct Array;

impl IndexAssign<(), ()> for Array {
    //~^ error: use of unstable library feature 'index_assign_trait'
    fn index_assign(&mut self, _: (), _: ()) {
        //~^ error: use of unstable library feature 'index_assign_trait'
        unimplemented!()
    }
}

fn main() {}

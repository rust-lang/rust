// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(experimental)];

use std::local_data;
use std::gc::{Gc, set_collection_frequency};

local_data_key!(GC_KEY: Gc<bool>)

fn main() {
    set_collection_frequency(1);
    // we squirrel away a GC pointer, and then check that it doesn't
    // get overwritten.
    local_data::set(GC_KEY, Gc::new(true));

    for _ in range(0, 20) {Gc::new(false);}

    local_data::get(GC_KEY, |ptr| assert!(unsafe {*ptr.unwrap().borrow()}));
}

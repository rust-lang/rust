// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators, generator_trait, box_leak)]

use std::cell::RefCell;
use std::ops::Generator;

fn main() {
    let (mut gen, cell);
    cell = Box::new(RefCell::new(0));
    let ref_ = Box::leak(Box::new(Some(cell.borrow_mut())));
    //~^ ERROR `*cell` does not live long enough [E0597]
    // the upvar is the non-dropck `&mut Option<Ref<'a, i32>>`.
    gen = || {
        // but the generator can use it to drop a `Ref<'a, i32>`.
        let _d = ref_.take(); //~ ERROR `ref_` does not live long enough
        yield;
    };
    unsafe { gen.resume(); }
    // drops the RefCell and then the Ref, leading to use-after-free
}

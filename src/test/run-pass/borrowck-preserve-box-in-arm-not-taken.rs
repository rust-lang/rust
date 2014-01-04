// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_POISON_ON_FREE=1

#[feature(managed_boxes)];

use std::cell::RefCell;

pub fn main() {
    let x: @RefCell<@Option<~int>> = @RefCell::new(@None);
    let mut xb = x.borrow_mut();
    match *xb.get() {
      @Some(ref _y) => {
        // here, the refcount of `*x` is bumped so
        // `_y` remains valid even if `*x` is modified.
        *xb.get() = @None;
      }
      @None => {
        // here, no bump of the ref count of `*x` is needed, but in
        // fact a bump occurs anyway because of how pattern marching
        // works.
      }
    }
}

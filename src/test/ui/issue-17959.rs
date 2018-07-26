// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate core;

use core::ops::Drop;

trait Bar {}

struct G<T: ?Sized> {
    _ptr: *const T
}

impl<T> Drop for G<T> {
//~^ ERROR: The requirement `T: std::marker::Sized` is added only by the Drop impl. [E0367]
    fn drop(&mut self) {
        if !self._ptr.is_null() {
        }
    }
}

fn main() {
    let x:G<Bar>;
}

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

extern crate debug;
use std::cell::Cell;

struct r {
  i: @Cell<int>,
}

#[unsafe_destructor]
impl Drop for r {
    fn drop(&mut self) {
        unsafe {
            self.i.set(self.i.get() + 1);
        }
    }
}

fn f<T>(_i: Vec<T> , _j: Vec<T> ) {
}

fn main() {
    let i1 = @Cell::new(0);
    let i2 = @Cell::new(1);
    let r1 = vec!(box r { i: i1 });
    let r2 = vec!(box r { i: i2 });
    f(r1.clone(), r2.clone());
    //~^ ERROR failed to find an implementation of
    println!("{:?}", (r2, i1.get()));
    println!("{:?}", (r1, i2.get()));
}

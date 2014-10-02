// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsafe_destructor)]

use std::cell::Cell;

struct shrinky_pointer<'a> {
  i: &'a Cell<int>,
}

#[unsafe_destructor]
impl<'a> Drop for shrinky_pointer<'a> {
    fn drop(&mut self) {
        println!("Hello!"); self.i.set(self.i.get() - 1);
    }
}

impl<'a> shrinky_pointer<'a> {
    pub fn look_at(&self) -> int { return self.i.get(); }
}

fn shrinky_pointer(i: &Cell<int>) -> shrinky_pointer {
    shrinky_pointer {
        i: i
    }
}

pub fn main() {
    let my_total = &Cell::new(10);
    { let pt = shrinky_pointer(my_total); assert!((pt.look_at() == 10)); }
    println!("my_total = {}", my_total.get());
    assert_eq!(my_total.get(), 9);
}

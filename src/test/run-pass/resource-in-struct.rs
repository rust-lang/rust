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

// Ensures that class dtors run if the object is inside an enum
// variant

use std::cell::Cell;

type closable<'a> = &'a Cell<bool>;

struct close_res<'a> {
  i: closable<'a>,

}

#[unsafe_destructor]
impl<'a> Drop for close_res<'a> {
    fn drop(&mut self) {
        self.i.set(false);
    }
}

fn close_res(i: closable) -> close_res {
    close_res {
        i: i
    }
}

enum option<T> { none, some(T), }

fn sink(_res: option<close_res>) { }

pub fn main() {
    let c = &Cell::new(true);
    sink(none);
    sink(some(close_res(c)));
    assert!(!c.get());
}

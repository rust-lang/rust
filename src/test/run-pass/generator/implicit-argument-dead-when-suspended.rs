// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators)]

use std::cell::Cell;

struct Flag<'a>(&'a Cell<bool>);

impl<'a> Drop for Flag<'a> {
    fn drop(&mut self) {
        self.0.set(false)
    }
}

fn main() {
    let alive = Cell::new(true);

    let gen = || {
        yield;
    };

    gen.resume(Flag(&alive));

    assert_eq!(alive.get(), false);
}

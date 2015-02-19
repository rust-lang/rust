// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

use std::cell::RefCell;

fn main() {
    let mut y = 1_usize;
    let c = RefCell::new(vec![]);
    c.push(box || y = 0);
    c.push(box || y = 0);
//~^ ERROR cannot borrow `y` as mutable more than once at a time
}

fn ufcs() {
    let mut y = 1_usize;
    let c = RefCell::new(vec![]);

    Push::push(&c, box || y = 0);
    Push::push(&c, box || y = 0);
//~^ ERROR cannot borrow `y` as mutable more than once at a time
}

trait Push<'c> {
    fn push<'f: 'c>(&self, push: Box<FnMut() + 'f>);
}

impl<'c> Push<'c> for RefCell<Vec<Box<FnMut() + 'c>>> {
    fn push<'f: 'c>(&self, fun: Box<FnMut() + 'f>) {
        self.borrow_mut().push(fun)
    }
}

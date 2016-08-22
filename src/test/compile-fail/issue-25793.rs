// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! width(
    ($this:expr) => {
        $this.width.unwrap()
        //~^ ERROR cannot use `self.width` because it was mutably borrowed
        //~| NOTE use of borrowed `*self`
    }
);

struct HasInfo {
    width: Option<usize>
}

impl HasInfo {
    fn get_size(&mut self, n: usize) -> usize {
        n
    }

    fn get_other(&mut self) -> usize {
        self.get_size(width!(self))
        //~^ NOTE in this expansion of width!
        //~| NOTE borrow of `*self` occurs here
    }
}

fn main() {}

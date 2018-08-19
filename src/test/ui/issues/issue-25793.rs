// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(rustc_attrs)]
macro_rules! width(
    ($this:expr) => {
        $this.width.unwrap()
        //~^ ERROR cannot use `self.width` because it was mutably borrowed
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
        let r = &mut *self;
        r.get_size(width!(self))
    }
    // Above is like `self.get_size(width!(self))`, but it
    // deliberately avoids NLL's two phase borrow feature.
}

fn main() { }

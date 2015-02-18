// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;

struct MyPtr<'a>(&'a mut usize);
impl<'a> Deref for MyPtr<'a> {
    type Target = usize;

    fn deref<'b>(&'b self) -> &'b usize { self.0 }
}

trait Tr {
    fn poke(&self, s: &mut usize);
}

impl Tr for usize {
    fn poke(&self, s: &mut usize)  {
        *s = 2;
    }
}

fn main() {
    let s = &mut 1_usize;

    MyPtr(s).poke(s);
    //~^ ERROR cannot borrow `*s` as mutable more than once at a time
}

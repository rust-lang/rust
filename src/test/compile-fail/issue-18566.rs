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

struct MyPtr<'a>(&'a mut uint);
impl<'a> Deref for MyPtr<'a> {
    type Target = uint;

    fn deref<'b>(&'b self) -> &'b uint { self.0 }
}

trait Tr {
    fn poke(&self, s: &mut uint);
}

impl Tr for uint {
    fn poke(&self, s: &mut uint)  {
        *s = 2;
    }
}

fn main() {
    let s = &mut 1u;

    MyPtr(s).poke(s);
    //~^ ERROR cannot borrow `*s` as mutable more than once at a time
}

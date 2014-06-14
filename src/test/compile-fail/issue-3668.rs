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

use std::gc::Gc;

struct P { child: Option<Gc<P>> }
trait PTrait {
   fn getChildOption(&self) -> Option<Gc<P>>;
}

impl PTrait for P {
   fn getChildOption(&self) -> Option<Gc<P>> {
       static childVal: Gc<P> = self.child.get();
       //~^ ERROR attempt to use a non-constant value in a constant
       fail!();
   }
}

fn main() {}

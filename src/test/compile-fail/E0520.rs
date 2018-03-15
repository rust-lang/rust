// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

trait SpaceLlama {
    fn fly(&self);
}

impl<T> SpaceLlama for T {
    default fn fly(&self) {}
}

impl<T: Clone> SpaceLlama for T {
//~^ NOTE parent `impl` is here
    fn fly(&self) {}
}

impl SpaceLlama for i32 {
    default fn fly(&self) {}
    //~^ ERROR E0520
    //~| NOTE cannot specialize default item `fly`
    //~| NOTE `fly` in the parent `impl` must be marked `default`
}

fn main() {
}

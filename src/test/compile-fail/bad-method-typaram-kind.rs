// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo<T:'static>() {
    1_usize.bar::<T>(); //~ ERROR `core::marker::Send` is not implemented
}

trait bar {
    fn bar<T:Send>(&self);
}

impl bar for usize {
    fn bar<T:Send>(&self) {
    }
}

fn main() {}

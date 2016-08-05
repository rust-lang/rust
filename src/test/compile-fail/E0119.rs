// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for T { //~ NOTE first implementation here
    fn get(&self) -> usize { 0 }
}

struct Foo {
    value: usize
}

impl MyTrait for Foo { //~ ERROR E0119
                       //~| NOTE conflicting implementation for `Foo`
    fn get(&self) -> usize { self.value }
}

fn main() {
}

// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait TheTrait : TheSuperTrait<<Self as TheTrait>::Item> {
    type Item;
}

trait TheSuperTrait<T> {
    fn get(&self) -> T;
}

impl TheTrait for i32 {
    type Item = u32;
}

impl TheSuperTrait<u32> for i32 {
    fn get(&self) -> u32 {
        *self as u32
    }
}

fn foo<T:TheTrait<Item=u32>>(t: &T) -> u32 {
    t.get()
}

fn main() {
    foo::<i32>(&22);
}

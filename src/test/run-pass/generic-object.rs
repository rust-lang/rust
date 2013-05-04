// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo<T> {
    fn get(&self) -> T;
}

struct S {
    x: int
}

impl Foo<int> for S {
    fn get(&self) -> int {
        self.x
    }
}

pub fn main() {
    let x = @S { x: 1 };
    let y = x as @Foo<int>;
    assert!(y.get() == 1);
}

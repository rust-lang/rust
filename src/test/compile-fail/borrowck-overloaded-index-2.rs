// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct MyVec<T> {
    data: Vec<T>,
}

impl<T> Index<uint, T> for MyVec<T> {
    fn index(&self, &i: &uint) -> &T {
        &self.data[i]
    }
}

fn main() {
    let v = MyVec { data: vec!(box 1i, box 2, box 3) };
    let good = &v[0]; // Shouldn't fail here
    let bad = v[0];
    //~^ ERROR cannot move out of dereference (dereference is implicit, due to indexing)
}

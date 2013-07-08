// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Clone2 {
    fn clone(&self) -> Self;
}


trait Getter<T: Clone2> {
    fn get(&self) -> T;
}

impl Getter<int> for int { //~ ERROR failed to find an implementation of trait Clone2 for int
    fn get(&self) -> int { *self }
}

fn main() { }

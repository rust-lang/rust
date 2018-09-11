// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

struct S<T> {
    contents: T,
}

impl<T> S<T> {
    fn new<U>(x: T, _: U) -> S<T> {
        S {
            contents: x,
        }
    }
}

trait Trait<T> {
    fn new<U>(x: T, y: U) -> Self;
}

struct S2 {
    contents: isize,
}

impl Trait<isize> for S2 {
    fn new<U>(x: isize, _: U) -> S2 {
        S2 {
            contents: x,
        }
    }
}

pub fn main() {
    let _ = S::<isize>::new::<f64>(1, 1.0);
    let _: S2 = Trait::<isize>::new::<f64>(1, 1.0);
}

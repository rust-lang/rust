// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait U {}
trait T<X: U> {}

trait S2<Y: U> {
    fn m(x: Box<T<Y>>) {}
}

struct St<X: U> {
    f: Box<T<X>>,
}

impl<X: U> St<X> {
    fn blah() {}
}

fn main() {}

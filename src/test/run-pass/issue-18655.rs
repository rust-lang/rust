// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Factory {
    type Product;
    fn create(&self) -> <Self as Factory>::Product;
}

impl Factory for f64 {
    type Product = f64;
    fn create(&self) -> f64 { *self * *self }
}

impl<A: Factory, B: Factory> Factory for (A, B) {
    type Product = (<A as Factory>::Product, <B as Factory>::Product);
    fn create(&self) -> (<A as Factory>::Product, <B as Factory>::Product) {
        let (ref a, ref b) = *self;
        (a.create(), b.create())
    }
}

fn main() {
    assert_eq!((16., 25.), (4., 5.).create());
}

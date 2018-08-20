// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo() {}
fn bar<T>() {}

fn main() {
    foo::<f64>(); //~ ERROR wrong number of type arguments: expected 0, found 1 [E0087]

    bar::<f64, u64>(); //~ ERROR wrong number of type arguments: expected 1, found 2 [E0087]
}

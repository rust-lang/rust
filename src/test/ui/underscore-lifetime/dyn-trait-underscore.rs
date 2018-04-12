// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the `'_` in `dyn Trait + '_` acts like ordinary elision,
// and not like an object lifetime default.
//
// cc #48468

fn a<T>(items: &[T]) -> Box<dyn Iterator<Item=&T>> {
    //                      ^^^^^^^^^^^^^^^^^^^^^ bound *here* defaults to `'static`
    Box::new(items.iter()) //~ ERROR cannot infer an appropriate lifetime
}

fn b<T>(items: &[T]) -> Box<dyn Iterator<Item=&T> + '_> {
    Box::new(items.iter()) // OK, equivalent to c
}

fn c<'a, T>(items: &'a [T]) -> Box<dyn Iterator<Item=&'a T> + 'a> {
    Box::new(items.iter()) // OK, equivalent to b
}

fn main() { }

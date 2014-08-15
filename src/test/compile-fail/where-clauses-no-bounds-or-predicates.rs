// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn equal1<T>(_: &T, _: &T) -> bool where {
//~^ ERROR a `where` clause must have at least one predicate in it
    true
}

fn equal2<T>(_: &T, _: &T) -> bool where T: {
//~^ ERROR each predicate in a `where` clause must have at least one bound
    true
}

fn main() {
}


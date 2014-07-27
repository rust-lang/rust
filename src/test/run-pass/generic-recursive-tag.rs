// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty FIXME(#14193)


use std::gc::{Gc, GC};

enum list<T> { cons(Gc<T>, Gc<list<T>>), nil, }

pub fn main() {
    let _a: list<int> =
        cons::<int>(box(GC) 10,
        box(GC) cons::<int>(box(GC) 12,
        box(GC) cons::<int>(box(GC) 13,
        box(GC) nil::<int>)));
}

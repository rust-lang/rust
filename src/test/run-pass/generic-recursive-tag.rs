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


enum list<T> { cons(Box<T>, Box<list<T>>), nil, }

pub fn main() {
    let _a: list<int> =
        cons::<int>(box 10,
        box cons::<int>(box 12,
        box cons::<int>(box 13,
        box nil::<int>)));
}

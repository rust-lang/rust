// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)
// min-lldb-version: 310

// compile-flags:-g

struct Peekable<I> where I: Iterator {
    _iter: I,
    _next: Option<<I as Iterator>::Item>,
}

fn main() {
    let mut iter = Vec::<i32>::new().into_iter();
    let next = iter.next();
    let _v = Peekable {
        _iter: iter,
        _next : next,
    };
}

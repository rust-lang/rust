// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type Big = [u64, ..8];
struct Pair<'self> { a: int, b: &'self Big }
static x: &'static Big = &([13, 14, 10, 13, 11, 14, 14, 15]);
static y: &'static Pair<'static> = &Pair {a: 15, b: x};

pub fn main() {
    assert!(ptr::to_unsafe_ptr(x) == ptr::to_unsafe_ptr(y.b));
}

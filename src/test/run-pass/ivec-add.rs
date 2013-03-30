// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn double<T:Copy>(a: T) -> ~[T] { return ~[a] + ~[a]; }

fn double_int(a: int) -> ~[int] { return ~[a] + ~[a]; }

pub fn main() {
    let mut d = double(1);
    assert!((d[0] == 1));
    assert!((d[1] == 1));

    d = double_int(1);
    assert!((d[0] == 1));
    assert!((d[1] == 1));
}


// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;
use core::task::spawn;

struct Pair {
    a: int,
    b: int
}

pub fn main() {
    let z = ~Pair { a : 10, b : 12};

    let f: ~fn() = || {
        assert!((z.a == 10));
        assert!((z.b == 12));
    };

    spawn(f);
}

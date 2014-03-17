// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task::spawn;

struct Pair {
    a: int,
    b: int
}

pub fn main() {
    let z = ~Pair { a : 10, b : 12};

    let f: proc() = proc() {
        assert_eq!(z.a, 10);
        assert_eq!(z.b, 12);
    };

    spawn(f);
}

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Pair { f: int, g: int }

pub fn main() {

    let x = Pair {
        f: 0,
        g: 0,
    };

    let _y = Pair {
        f: 1,
        g: 1,
        .. x
    };

    let _z = Pair {
        f: 1,
        .. x
    };

}

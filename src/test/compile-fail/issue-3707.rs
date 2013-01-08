// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
struct Obj {
    member: uint
}

impl Obj {
    static pure fn boom() -> bool {
        return 1+1 == 2
    }
    pure fn chirp() {
        self.boom(); //~ ERROR wat
    }
}

fn main() {
    let o = Obj { member: 0 };
    o.chirp();
    1 + 1;
}

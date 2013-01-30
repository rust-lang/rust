// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:unused import
// compile-flags:-W unused-imports
use cal = bar::c::cc;

mod foo {
    pub type point = {x: int, y: int};
    pub type square = {p: point, h: uint, w: uint};
}

mod bar {
    pub mod c {
        use foo::point;
        use foo::square;
        pub fn cc(p: point) -> str { return 2 * (p.x + p.y); }
    }
}

fn main() {
    cal({x:3, y:9});
}

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -D unused-imports

use cal = bar::c::cc;

use core::either::Right;        //~ ERROR unused import

use core::util::*;              // shouldn't get errors for not using
                                // everything imported

// Should only get one error instead of two errors here
use core::option::{Some, None}; //~ ERROR unused import

use core::io::ReaderUtil;       //~ ERROR unused import
// Be sure that if we just bring some methods into scope that they're also
// counted as being used.
use core::io::WriterUtil;

mod foo {
    pub struct Point{x: int, y: int}
    pub struct Square{p: Point, h: uint, w: uint}
}

mod bar {
    pub mod c {
        use foo::Point;
        use foo::Square; //~ ERROR unused import
        pub fn cc(p: Point) -> int { return 2 * (p.x + p.y); }
    }
}

fn main() {
    cal(foo::Point{x:3, y:9});
    let a = 3;
    ignore(a);
    io::stdout().write_str(~"a");
}

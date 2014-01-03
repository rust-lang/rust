// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(globs)];
#[deny(unused_imports)];
#[allow(dead_code)];

use cal = bar::c::cc;

use std::util::*;              // shouldn't get errors for not using
                                // everything imported

// Should get errors for both 'Some' and 'None'
use std::option::{Some, None}; //~ ERROR unused import
                                //~^ ERROR unused import

use test::A;       //~ ERROR unused import
// Be sure that if we just bring some methods into scope that they're also
// counted as being used.
use test::B;

// Make sure this import is warned about when at least one of its imported names
// is unused
use std::vec::{from_fn, from_elem};   //~ ERROR unused import

mod test {
    pub trait A { fn a(&self) {} }
    pub trait B { fn b(&self) {} }
    pub struct C;
    impl A for C {}
    impl B for C {}
}

mod foo {
    pub struct Point{x: int, y: int}
    pub struct Square{p: Point, h: uint, w: uint}
}

mod bar {
    // Don't ignore on 'pub use' because we're not sure if it's used or not
    pub use std::cmp::Eq;

    pub mod c {
        use foo::Point;
        use foo::Square; //~ ERROR unused import
        pub fn cc(p: Point) -> int { return 2 * (p.x + p.y); }
    }

    #[allow(unused_imports)]
    mod foo {
        use std::cmp::Eq;
    }
}

fn main() {
    cal(foo::Point{x:3, y:9});
    let a = 3;
    id(a);
    test::C.b();
    let _a = from_elem(0, 0);
}

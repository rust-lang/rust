// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #32326. We ran out of memory because we
// attempted to expand this case up to the recursion limit, and 2^N is
// too big.

enum Expr { //~ ERROR E0072
            //~| NOTE recursive type has infinite size
    Plus(Expr, Expr),
    Literal(i64),
}

fn main() { }

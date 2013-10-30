// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    1 = 2; //~ ERROR illegal left-hand side expression
    1 += 2; //~ ERROR illegal left-hand side expression
    (1, 2) = (3, 4); //~ ERROR illegal left-hand side expression

    let (a, b) = (1, 2);
    (a, b) = (3, 4); //~ ERROR illegal left-hand side expression

    None = Some(3); //~ ERROR illegal left-hand side expression
}

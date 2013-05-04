// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct point { x: int, y: int }

fn a() {
    let mut p = point {x: 3, y: 4};
    let _q = &p; //~ NOTE loan of mutable local variable granted here

    // This assignment is illegal because the field x is not
    // inherently mutable; since `p` was made immutable, `p.x` is now
    // immutable.  Otherwise the type of &_q.x (&int) would be wrong.
    p.x = 5; //~ ERROR assigning to mutable field prohibited due to outstanding loan
}

fn c() {
    // this is sort of the opposite.  We take a loan to the interior of `p`
    // and then try to overwrite `p` as a whole.

    let mut p = point {x: 3, y: 4};
    let _q = &p.y; //~ NOTE loan of mutable local variable granted here
    p = point {x: 5, y: 7};//~ ERROR assigning to mutable local variable prohibited due to outstanding loan
    copy p;
}

fn d() {
    // just for completeness's sake, the easy case, where we take the
    // address of a subcomponent and then modify that subcomponent:

    let mut p = point {x: 3, y: 4};
    let _q = &p.y; //~ NOTE loan of mutable field granted here
    p.y = 5; //~ ERROR assigning to mutable field prohibited due to outstanding loan
    copy p;
}

fn main() {
}

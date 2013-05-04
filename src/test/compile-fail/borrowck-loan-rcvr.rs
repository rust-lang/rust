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

trait methods {
    fn impurem(&self);
    fn blockm(&self, f: &fn());
    fn purem(&self);
}

impl methods for point {
    fn impurem(&self) {
    }

    fn blockm(&self, f: &fn()) { f() }

    fn purem(&self) {
    }
}

fn a() {
    let mut p = point {x: 3, y: 4};

    // Here: it's ok to call even though receiver is mutable, because we
    // can loan it out.
    p.purem();
    p.impurem();

    // But in this case we do not honor the loan:
    do p.blockm { //~ NOTE loan of mutable local variable granted here
        p.x = 10; //~ ERROR assigning to mutable field prohibited due to outstanding loan
    }
}

fn b() {
    let mut p = point {x: 3, y: 4};

    // Here I create an outstanding loan and check that we get conflicts:

    let l = &mut p; //~ NOTE prior loan as mutable granted here
    //~^ NOTE prior loan as mutable granted here

    p.purem(); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
    p.impurem(); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan

    l.x += 1;
}

fn c() {
    // Loaning @mut as & is considered legal due to dynamic checks:
    let q = @mut point {x: 3, y: 4};
    q.purem();
    q.impurem();
}

fn main() {
}

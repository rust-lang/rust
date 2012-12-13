// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let d = dvec::DVec();
    d.push(3);
    d.push(4);
    assert d.get() == ~[3, 4];
    d.set(~[5]);
    d.push(6);
    d.push(7);
    d.push(8);
    d.push(9);
    d.push(10);
    d.push_all(~[11, 12, 13]);
    d.push_slice(~[11, 12, 13], 1u, 2u);

    let exp = ~[5, 6, 7, 8, 9, 10, 11, 12, 13, 12];
    assert d.get() == exp;
    assert d.get() == exp;
    assert d.len() == exp.len();

    for d.eachi |i, e| {
        assert *e == exp[i];
    }

    let v = dvec::unwrap(move d);
    assert v == exp;
}

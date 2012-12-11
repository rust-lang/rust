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
    match {a: 10, b: @20} {
        x@{a, b: @20} => { assert x.a == 10; assert a == 10; }
        {b, _} => { fail; }
    }
    let x@{b, _} = {a: 10, b: {mut c: 20}};
    x.b.c = 30;
    assert b.c == 20;
    let y@{d, _} = {a: 10, d: {mut c: 20}};
    y.d.c = 30;
    assert d.c == 20;
}

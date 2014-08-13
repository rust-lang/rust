// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #16205.

struct Foo {
    a: [Box<int>, ..3],
}

fn main() {
    let mut y = 1i;
    let x = Some(&mut y);
    for &a in x.iter() {    //~ ERROR cannot move out
    }

    let f = Foo {
        a: [box 3, box 4, box 5],
    };
    for &a in f.a.iter() {  //~ ERROR cannot move out
    }

    let x = Some(box 1i);
    for &a in x.iter() {    //~ ERROR cannot move out
    }
}

